import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, Any


def crop_and_format_phase_map(calibrated_phase: np.ndarray, nozzle_edge_y: int, px_to_mm: float,
                              max_height_mm: float = 6.0) -> np.ndarray:
    """
    Crops the phase map to only include the region from the nozzle edge up to a specified height.
    Flips the array so that row 0 corresponds exactly to the nozzle edge.

    Args:
        calibrated_phase: The full 2D phase map.
        nozzle_edge_y: The pixel index of the nozzle edge.
        px_to_mm: Spatial calibration factor.
        max_height_mm: How many millimeters above the nozzle to keep.

    Returns:
        np.ndarray: The cropped and flipped phase map.
    """
    # Calculate how many pixels correspond to the max height
    height_px = int(max_height_mm / px_to_mm)

    # Define the crop bounds (y=0 is the top of the raw image, nozzle points up)
    y_end = int(nozzle_edge_y)
    y_start = max(0, y_end - height_px)

    # Crop to just the gas jet area
    phi_cropped = calibrated_phase[y_start:y_end, :]

    # Flip upside down so that index 0 is the nozzle edge and it goes UP in height
    return np.flipud(phi_cropped)


def estimate_depth_profile(
        phi_cropped: np.ndarray,
        px_to_mm: float,
        nozzle_opening: float,  # <--- NEW: Physical width of the nozzle
        use_abs: bool = False,
        smooth_win_px: int = 9,
        min_peak_abs: float = 0.0,
        bg_frac: float = 0.10,
        exclude_y_mm: float = 0.4,
        fit_type: str = "quadratic_constrained",
        # Options: "quadratic_constrained", "linear_constrained", "unconstrained"
        debug_y_mm: Optional[float] = None,
        debug_row_idx: Optional[int] = None,
        debug_title: Optional[str] = None,
        debug_show: bool = False,
        wrap_jump_thr: float = 1.6 * np.pi,
        unwrap_rows: bool = True,
        fwhm_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute an effective 'depth' profile d(y) from a properly cropped/flipped phase map.
    Anchors the depth at y=0 to the physical nozzle_width_mm to prevent negative depths.
    """
    sig = np.abs(phi_cropped) if use_abs else phi_cropped
    h, w = sig.shape

    y_mm = np.arange(h, dtype=np.float32) * px_to_mm  # 0 at nozzle edge

    # --- allocate outputs ---
    depth_mm = np.full(h, np.nan, dtype=np.float32)
    left_idx = np.full(h, -1, dtype=np.int32)
    right_idx = np.full(h, -1, dtype=np.int32)
    peak_val = np.full(h, np.nan, dtype=np.float32)
    bg_val = np.full(h, np.nan, dtype=np.float32)
    thr_val = np.full(h, np.nan, dtype=np.float32)
    peak_idx = np.full(h, -1, dtype=np.int32)

    n_bg = max(1, int(bg_frac * w))
    left_bg = slice(0, n_bg)
    right_bg = slice(w - n_bg, w)

    dbg_i: Optional[int] = None
    if debug_row_idx is not None:
        dbg_i = int(debug_row_idx)
    elif debug_y_mm is not None:
        dbg_i = int(np.clip(round(float(debug_y_mm) / px_to_mm), 0, h - 1))

    debug_row = {}
    if unwrap_rows:
        sig_unwrapped = np.unwrap(sig)

    for i in range(h):
        row = sig[i].astype(np.float32)
        bg = np.nanmedian(np.concatenate([row[left_bg], row[right_bg]]))
        bg_val[i] = float(bg)
        row_work = row - bg

        y_loc = y_mm[i]
        is_close_to_nozzle = y_loc <= 0.2

        if not is_close_to_nozzle:
            if unwrap_rows:
                d = np.diff(row_work)
                if np.any(np.abs(d) > float(wrap_jump_thr)):
                    row_work = np.unwrap(row_work).astype(np.float32)
                    bg2 = np.nanmedian(np.concatenate([row_work[left_bg], row_work[right_bg]]))
                    row_work = row_work - bg2

            if smooth_win_px is not None and smooth_win_px > 1:
                k = int(smooth_win_px)
                kernel = np.ones(k, dtype=np.float32) / k
                row_work = np.convolve(row_work, kernel, mode="same")

        pmax_raw = float(np.nanmax(row_work))
        pmin_raw = float(np.nanmin(row_work))

        if abs(pmin_raw) > abs(pmax_raw):
            row0 = -row_work
            pmax = abs(pmin_raw)
        else:
            row0 = row_work
            pmax = pmax_raw

        peak_val[i] = pmax
        imax = int(np.nanargmax(row0))
        peak_idx[i] = imax

        if is_close_to_nozzle:
            noise_std = np.nanstd(np.concatenate([row_work[left_bg], row_work[right_bg]]))
            thr = 30.0 * noise_std
            mask = abs(row0) >= thr
        else:
            thr = float(fwhm_frac * pmax)
            mask = row0 >= thr

        thr_val[i] = float(thr)

        if not np.isfinite(pmax) or pmax <= float(min_peak_abs) or not np.any(mask):
            continue

        if is_close_to_nozzle:
            l = np.where(mask)[0][0]
            r = np.where(mask)[0][-1]
            l, r = max(0, l + 1), min(w - 1, r - 1)
            left_idx[i], right_idx[i] = l, r
            depth_mm[i] = (r - l + 1) * px_to_mm
        else:
            def gaussian_1d(x, amplitude, mean, sigma):
                return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

            try:
                x_indices = np.arange(w, dtype=np.float32)
                row0_for_fit = row0.copy()
                if row0_for_fit[imax] < 0:
                    row0_for_fit = -row0_for_fit

                pmax_abs = float(np.abs(np.nanmax(row0_for_fit)))
                x_peak = float(imax)

                if np.any(mask):
                    mask_indices = np.where(mask)[0]
                    fwhm_estimate = max(1.0, float(mask_indices[-1] - mask_indices[0]))
                    sigma_init = fwhm_estimate / 2.355
                else:
                    sigma_init = 2.0

                p0 = [pmax_abs, x_peak, sigma_init]
                bounds = ([0.1 * pmax_abs, x_peak - w / 2, 0.5], [2.0 * pmax_abs, x_peak + w / 2, w])

                coeffs, _ = curve_fit(gaussian_1d, x_indices, row0_for_fit, p0=p0, bounds=bounds, maxfev=5000)
                amplitude, mean, sigma = coeffs

                gaussian_integral = float(sigma) * np.sqrt(2.0 * np.pi)
                depth_mm[i] = gaussian_integral * px_to_mm

            except Exception as e:
                l, r = imax, imax
                while l > 0 and mask[l]: l -= 1
                while r < w - 1 and mask[r]: r += 1
                l, r = max(0, l + 1), min(w - 1, r - 1)
                left_idx[i], right_idx[i] = l, r
                depth_mm[i] = (r - l + 1) * px_to_mm

    # =========================================================================
    # NEW: CONSTRAINED FIT ANCHORED AT NOZZLE WIDTH
    # =========================================================================
    valid_mask = np.isfinite(depth_mm) & (depth_mm > 0) & (y_mm > exclude_y_mm)

    if np.sum(valid_mask) < 3:
        print(f"Warning: Not enough valid points above {exclude_y_mm}mm to fit depth.")
        valid_mask = np.isfinite(depth_mm) & (depth_mm > 0)

    y_fit = y_mm[valid_mask]
    d_fit = depth_mm[valid_mask]

    if fit_type == "quadratic_constrained":
        # Fits a curve but forces the depth at y=0 to be exactly nozzle_width_mm
        def model(y, a, b):
            return a * (y ** 2) + b * y + nozzle_opening

        try:
            coeffs, _ = curve_fit(model, y_fit, d_fit)
            depth_fit = model(y_mm, *coeffs)
            fit_label = f"Quadratic fit (anchored at {nozzle_opening}mm)"
        except Exception:
            fit_type = "linear_constrained"

    if fit_type == "linear_constrained":
        # Fits a line but forces the depth at y=0 to be exactly nozzle_width_mm
        def model(y, slope):
            return slope * y + nozzle_opening

        coeffs, _ = curve_fit(model, y_fit, d_fit)
        depth_fit = model(y_mm, *coeffs)
        fit_label = f"Linear fit (anchored at {nozzle_opening}mm)"

    elif fit_type == "unconstrained":
        # Old method: simple linear fit, ignoring nozzle width
        coeffs = np.polyfit(y_fit, d_fit, 1)
        depth_fit = (coeffs[0] * y_mm) + coeffs[1]
        fit_label = "Unconstrained linear fit"

    # Safety catch: gas jet cannot be narrower than the physical nozzle itself
    depth_fit = np.maximum(depth_fit, nozzle_opening)

    debug = {
        "px_mm": px_to_mm,
        "nozzle_width_mm": nozzle_opening,
        "exclude_y_mm": exclude_y_mm,
    }

    if debug_show:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        ax.plot(depth_mm, y_mm, "o", color='lightgray', alpha=0.5, label="Raw depth estimates (noisy near nozzle)")
        ax.plot(d_fit, y_fit, ".", color='blue', alpha=0.8, label=f"Valid points used for fit (>{exclude_y_mm}mm)")
        ax.plot(depth_fit, y_mm, "-k", linewidth=2, label=fit_label)
        ax.axhline(exclude_y_mm, linestyle="--", color="red", alpha=0.5, label="Exclusion Zone Boundary")

        # Mark the physical nozzle width
        ax.plot(nozzle_opening, 0, 'r*', markersize=12, label="Physical Nozzle Width")

        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Height above nozzle (mm)")
        ax.set_title("Gas Jet Depth Profile")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.show()

    return y_mm, depth_fit, debug


def phase_to_density(
        phi_cropped: np.ndarray,
        px_to_mm: float,
        depth_y_mm: np.ndarray,
        depth_profile_mm: np.ndarray,
        gas: str = "argon"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a properly oriented phase map to number density.
    Interpolates the depth profile to match the y-coordinates of the phase map.
    """
    wavelength_m = 532e-9
    N_A = 6.022e23

    if gas.lower() == "argon":
        K = 1.57e-4
        M = 39.95e-3
    elif gas.lower() == "helium":
        K = 1.96e-4
        M = 4.0026e-3
    else:
        raise ValueError("gas must be 'argon' or 'helium'")

    # Spatial coordinates for the perpendicular phase map
    y_mm = np.arange(phi_cropped.shape[0]) * px_to_mm
    x_mm = np.arange(phi_cropped.shape[1]) * px_to_mm

    # Interpolate the depth profile onto the density map's y-grid
    # This safely handles cases where the 0-deg and 90-deg images have different row counts
    depth_interp_mm = np.interp(y_mm, depth_y_mm, depth_profile_mm, left=np.nan, right=np.nan)

    depth_m = depth_interp_mm * 1e-3

    # Avoid division by zero warnings
    depth_m_safe = np.where(depth_m <= 0, np.nan, depth_m)

    # phase -> mass density [kg/m^3]
    rho = (phi_cropped * wavelength_m) / (2 * np.pi * K * depth_m_safe[:, None])

    # mass density -> number density [cm^-3]
    density = (rho / M) * N_A / 1e6
    density = np.maximum(density, 0)

    return density, x_mm, y_mm