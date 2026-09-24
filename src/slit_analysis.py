from typing import Dict, List, Tuple, Optional, Sequence, Any, Union
import numpy as np
from matplotlib import pyplot as plt

def compute_depth_profile(
        phi: np.ndarray,
        *,
        width_mm: float = 5,
        use_abs: bool = False,
        smooth_win_px: int = 9,
        min_peak_abs: float = 0.0,
        tol: Optional[float] = None,
        nozzle_row: str = "last",  # "last" if nozzle is at bottom in raw saved array
        bg_frac: float = 0.10,  # background from outer x edges
        debug_y_mm: Optional[float] = None,
        debug_row_idx: Optional[int] = None,
        debug_title: Optional[str] = None,
        debug_show: bool = False,
        wrap_jump_thr: float = 1.6 * np.pi,
        unwrap_rows: bool = True,
        fwhm_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """
    Compute an effective 'depth' profile d(y) from the PARALLEL orientation phase map
    using Gaussian fitting and integration.

    Physical Model:
      - Assumes ρ(x,y,z) = G(x,z) * ρ(y,z) where G is Gaussian in x with amplitude ~1
      - Phase integral: φ(x,y,z) ~ κ * ρ(y,z) * ∫G(x)dx
      - For normalized Gaussian: ∫G(x)dx = σ√(2π)

    Method:
      1. For each row (y position), fit a Gaussian to the phase profile
      2. Normalize amplitude to 1 and extract σ (width parameter)
      3. Compute effective depth as σ√(2π) * px_mm

    This replaces the old FWHM rectangular approximation with proper Gaussian integration.

    Debugging:
      - set debug_y_mm (preferred) or debug_row_idx to plot the chosen row and show
        the fitted Gaussian and integration result.

    Returns:
      y_mm, depth_mm, entry, debug
    """


    # Try to load metadata for accurate scaling
    if metadata:
        # Use actual roi_width_mm from metadata
        actual_width_mm = metadata['roi_width_mm']
        h, w = phi.shape
        px_mm = float(actual_width_mm) / float(w)
    else:
        # Fallback to parameter (legacy phase maps)
        h, w = phi.shape
        px_mm = float(width_mm) / float(w)

    # --- enforce y convention: row 0 = nozzle edge -------------------------
    if nozzle_row == "last":
        phi_proc = np.flipud(phi)
    elif nozzle_row == "first":
        phi_proc = phi
    else:
        raise ValueError("nozzle_row must be 'first' or 'last'")

    sig = np.abs(phi_proc) if use_abs else phi_proc
    h, w = sig.shape

    y_mm = np.arange(h, dtype=np.float32) * px_mm  # 0 at nozzle edge

    # # --- smoothing along x (row-wise) --------------------------------------
    # if smooth_win_px is not None and smooth_win_px > 1:
    #     k = int(smooth_win_px)
    #     kernel = np.ones(k, dtype=np.float32) / k
    #     sig_s = np.empty_like(sig)
    #     for i in range(h):
    #         sig_s[i] = np.convolve(sig[i], kernel, mode="same")
    # else:
    #     sig_s = sig

    # --- allocate outputs ---------------------------------------------------
    depth_mm = np.full(h, np.nan, dtype=np.float32)
    left_idx = np.full(h, -1, dtype=np.int32)
    right_idx = np.full(h, -1, dtype=np.int32)
    peak_val = np.full(h, np.nan, dtype=np.float32)
    bg_val = np.full(h, np.nan, dtype=np.float32)
    thr_val = np.full(h, np.nan, dtype=np.float32)
    peak_idx = np.full(h, -1, dtype=np.int32)

    # background from edges
    n_bg = max(1, int(bg_frac * w))
    left_bg = slice(0, n_bg)
    right_bg = slice(w - n_bg, w)

    # choose debug row
    dbg_i: Optional[int] = None
    if debug_row_idx is not None:
        dbg_i = int(debug_row_idx)
    elif debug_y_mm is not None:
        dbg_i = int(np.clip(round(float(debug_y_mm) / px_mm), 0, h - 1))

    # keep row debug info
    debug_row = {}
    if unwrap_rows:
        sig_unwrapped = np.unwrap(sig)

    for i in range(h):
        row = sig[i].astype(np.float32)

        # 1. Baseline
        bg = np.nanmedian(np.concatenate([row[left_bg], row[right_bg]]))
        bg_val[i] = float(bg)
        row_work = row - bg

        y_loc = y_mm[i]
        is_close_to_nozzle = y_loc <= 0.2

        # 2. Phase Correction / Smoothing (Only for Far-Field or as needed)
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

        # 3. Determine Peak and Polarity
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

        # 4. Hybrid Masking Logic
        if is_close_to_nozzle:
            # Support detection: Use Noise Floor
            noise_std = np.nanstd(np.concatenate([row_work[left_bg], row_work[right_bg]]))
            thr = 30.0 * noise_std
            mask = abs(row0) >= thr
        else:
            # Standard FWHM
            thr = float(fwhm_frac * pmax)
            mask = row0 >= thr

        thr_val[i] = float(thr)

        # 5. Validity Check
        if not np.isfinite(pmax) or pmax <= float(min_peak_abs) or not np.any(mask):
            if dbg_i is not None and i == dbg_i:
                debug_row = {
                    "i": i, "y_mm": float(y_mm[i]), "row_raw": row,
                    "bg": float(bg), "row0": row0, "pmax": float(pmax),
                    "reason": "Signal below threshold or mask empty",
                }
            continue

        # 6. Width/Depth Calculation
        if is_close_to_nozzle:
            # Close to nozzle: Use simple edge-finding method (no Gaussian fitting)
            l = np.where(mask)[0][0]
            r = np.where(mask)[0][-3]
            l, r = max(0, l + 1), min(w - 1, r - 1)
            left_idx[i], right_idx[i] = l, r
            depth_mm[i] = (r - l + 1) * px_mm

            if dbg_i is not None and i == dbg_i:
                debug_row = {
                    "i": i, "y_mm": float(y_mm[i]), "row_raw": row, "bg": float(bg),
                    "row0": row0, "pmax": float(pmax), "thr": float(thr),
                    "mask": mask, "l": l, "r": r, "depth_mm": float(depth_mm[i]),
                    "method": "simple_width_near_nozzle"
                }
        else:
            # Far from nozzle: Use Gaussian fitting for proper integral calculation
            # Instead of FWHM width, fit a Gaussian and use its integral ∫G(x)dx = σ√(2π)
            # This correctly implements ρ(x,y,z) = G(z) * ρ(y,z)

            # Initialize base debug_row with required fields (needed for both success and failure paths)
            if dbg_i is not None and i == dbg_i:
                debug_row = {
                    "i": i,
                    "y_mm": float(y_mm[i]),
                    "row_raw": row,
                    "bg": float(bg),
                    "row0": row0,
                    "pmax": float(pmax),
                    "thr": float(thr),
                    "mask": mask,
                    "imax": imax,
                    "method": "gaussian_fit_far_field"
                }

            def gaussian_1d(x, amplitude, mean, sigma):
                """Standard 1D Gaussian"""
                return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

            try:
                x_indices = np.arange(w, dtype=np.float32)

                # Ensure row0 at peak has correct polarity (positive)
                # If row0[imax] is negative, flip row0 to make it positive
                row0_for_fit = row0.copy()
                if row0_for_fit[imax] < 0:
                    row0_for_fit = -row0_for_fit

                # Use absolute value of pmax to ensure bounds are always positive
                pmax_abs = float(np.abs(np.nanmax(row0_for_fit)))

                # Initial parameter guesses
                x_peak = float(imax)
                # Estimate sigma from FWHM ≈ 2.355 * sigma, using the mask width as proxy
                if np.any(mask):
                    mask_indices = np.where(mask)[0]
                    fwhm_estimate = max(1.0, float(mask_indices[-1] - mask_indices[0]))
                    sigma_init = fwhm_estimate / 2.355
                else:
                    sigma_init = 2.0

                # Fit Gaussian to row0 (baseline-subtracted)
                p0 = [pmax_abs, x_peak, sigma_init]

                # Perform fit with tight constraints (always positive amplitude)
                bounds = (
                    [0.1 * pmax_abs, x_peak - w/ 2, 0.5],  # lower bounds (always positive)
                    [2.0 * pmax_abs, x_peak + w / 2, w]  # upper bounds
                )

                coeffs, _ = curve_fit(gaussian_1d, x_indices, row0_for_fit, p0=p0, bounds=bounds, maxfev=5000)
                amplitude, mean, sigma = coeffs

                # Normalize amplitude to 1 and compute integral ∫G(x)dx = σ√(2π)
                # (This represents the effective thickness assuming amplitude-normalized Gaussian)
                gaussian_integral = float(sigma) * np.sqrt(2.0 * np.pi)
                depth_mm[i] = gaussian_integral * px_mm

                # Store fitting params in debug info
                if dbg_i is not None and i == dbg_i:
                    debug_row["gaussian_fit"] = {
                        "amplitude": float(amplitude),
                        "mean": float(mean),
                        "sigma": float(sigma),
                        "integral_px": float(gaussian_integral),
                        "depth_mm": float(depth_mm[i]),
                    }

            except Exception as e:
                # If fit fails, fall back to contiguous region search
                l, r = imax, imax
                while l > 0 and mask[l]: l -= 1
                while r < w - 1 and mask[r]: r += 1

                l, r = max(0, l + 1), min(w - 1, r - 1)
                left_idx[i], right_idx[i] = l, r
                depth_mm[i] = (r - l + 1) * px_mm

                if dbg_i is not None and i == dbg_i:
                    debug_row["gaussian_fit"] = {
                        "status": "failed - fallback to FWHM",
                        "error": str(e),
                        "depth_mm": float(depth_mm[i]),
                    }
    debug = {
        "matched_delay": entry.delay,
        "file_path": str(entry.path),
        "px_mm": px_mm,
        "width_mm": float(width_mm),
        "smooth_win_px": int(smooth_win_px),
        "use_abs": bool(use_abs),
        "min_peak_abs": float(min_peak_abs),
        "nozzle_row": nozzle_row,
        "bg_frac": float(bg_frac),
        "left_idx": left_idx,
        "right_idx": right_idx,
        "peak_idx": peak_idx,
        "peak_val": peak_val,
        "bg_val": bg_val,
        "thr_val": thr_val,
        "debug_row": debug_row,
    }

    # --- debug plot ---------------------------------------------------------
    if debug_show and debug_row:
        i = debug_row["i"]
        x_mm = np.arange(w, dtype=np.float32) * px_mm
        row = debug_row["row_raw"]
        row0 = debug_row["row0"]

        # Font sizes for paper figure
        label_fs = 16
        tick_fs = 14
        legend_fs = 13
        title_fs = 17
        panel_fs = 18

        fig, ax = plt.subplots(
            2, 1,
            figsize=(10, 7),
            sharex=True,
            constrained_layout=True
        )

        # ---------------- Panel 1 ----------------
        ax[0].plot(x_mm, row, linewidth=2, label="row (raw)")
        ax[0].axhline(
            debug_row["bg"],
            linestyle="--",
            linewidth=2,
            label=f"bg={debug_row['bg']:.3g}"
        )

        ax[0].set_ylabel(
            "signal (abs phase)" if use_abs else "signal (phase)",
            fontsize=label_fs
        )

        ax[0].legend(loc="best", fontsize=legend_fs)
        ax[0].tick_params(axis="both", labelsize=tick_fs)
        ax[0].grid(True, alpha=0.3)

        # Panel number
        ax[0].text(
            0.02, 0.95, "1",
            transform=ax[0].transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="top",
            ha="left"
        )

        # ---------------- Panel 2 ----------------
        ax[1].plot(x_mm, row0, linewidth=2, label="row0 = row - bg")

        if "thr" in debug_row:
            ax[1].axhline(
                debug_row["thr"],
                linestyle="--",
                linewidth=2,
                label=f"half-max thr={debug_row['thr']:.3g}"
            )

        if "imax" in debug_row:
            imax = debug_row["imax"]
            ax[1].axvline(
                x_mm[imax],
                linestyle=":",
                linewidth=2,
                label="peak"
            )

        if "l" in debug_row and "r" in debug_row:
            l, r = debug_row["l"], debug_row["r"]

            ax[1].axvline(
                x_mm[l],
                linestyle="--",
                linewidth=2,
                label="left FWHM"
            )
            ax[1].axvline(
                x_mm[r],
                linestyle="--",
                linewidth=2,
                label="right FWHM"
            )

            ax[1].fill_between(
                x_mm,
                0,
                row0,
                where=debug_row["mask"],
                alpha=0.2,
                step="mid",
                label="mask (>=thr)",
            )

        # Plot fitted Gaussian if available
        if (
                "gaussian_fit" in debug_row
                and "amplitude" in debug_row["gaussian_fit"]
        ):
            gfit = debug_row["gaussian_fit"]
            amp = gfit["amplitude"]
            mean = gfit["mean"]
            sigma = gfit["sigma"]

            x_indices = np.arange(len(row0))
            gaussian_fit_curve = amp * np.exp(
                -((x_indices - mean) ** 2) / (2 * sigma ** 2)
            )

            ax[1].plot(
                x_mm,
                gaussian_fit_curve,
                "r-",
                linewidth=2.5,
                label=f"Gaussian fit ($\\sigma$={sigma:.2f}px)"
            )

            ax[1].axvline(
                x_mm[int(mean)],
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label="fit mean"
            )

        # Panel number
        ax[1].text(
            0.02, 0.95, "2",
            transform=ax[1].transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="top",
            ha="left"
        )

        ttl = debug_title or (
            f"FWHM debug — delay={entry.delay:.6f}, "
            f"y={y_mm[i]:.3f} mm, row={i}"
        )

        ax[0].set_title(ttl, fontsize=title_fs)

        ax[1].set_xlabel("x (mm)", fontsize=label_fs)
        ax[1].set_ylabel("row0", fontsize=label_fs)

        ax[1].legend(loc="best", fontsize=legend_fs)
        ax[1].tick_params(axis="both", labelsize=tick_fs)
        ax[1].grid(True, alpha=0.3)

        plt.show()  # plot depth_mm vs y_mm
    elif debug_show:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        ax.plot(y_mm, depth_mm, label="depth profile")
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("depth (mm)")
        ax.set_title(f"Depth profile — delay={entry.delay:.6f}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.show()

    return y_mm, depth_mm, entry, debug

