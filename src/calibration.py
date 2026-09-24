from doctest import debug

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Sequence, Any, Union

def estimate_nozzle_edge(img, center_frac=(0.35, 0.7)):
    h, w = img.shape
    x1 = int(w * center_frac[0])
    x2 = int(w * center_frac[1])

    prof = img[:, x1:x2].mean(axis=1).astype(np.float32)

    win = max(51, h // 50)  # coarse window
    baseline = np.convolve(prof, np.ones(win) / win, mode="same")
    p = prof - baseline

    F = np.fft.rfft(p - p.mean())
    mag = np.abs(F)
    mag[:3] = 0 # ignore DC and very low frequencies
    k = np.argmax(mag)
    if k == 0:
        return None, None, None, {"prof": prof, "p": p, "baseline": baseline, "mag": mag}

    period_px = h / k
    min_dist = int(max(5, 0.7 * period_px))

    peaks, props = find_peaks(p, distance=min_dist, prominence=0)
    prominences = props.get("prominences", np.array([], dtype=np.float32))

    top_mask = peaks < int(0.25 * h)
    if np.any(top_mask):
        ref_prom = float(np.median(prominences[top_mask]))
    else:
        ref_prom = float(np.median(prominences))

    # plt.figure()
    # plt.plot(p)

    nozzle_edge_y = None
    for i in range(len(peaks) - 1, 0, -1):
        if prominences[i] > 0.2 * ref_prom:
            nozzle_edge_y = int(peaks[i] + 0.5 * (peaks[i] - peaks[i - 1]))

            # plt.plot(peaks[i], p[peaks[i]], "x")
            # plt.show()
            break
    debug = {
        "prof": prof,
        "baseline": baseline,
        "p": p,
        "mag": mag,
        "peaks": peaks,
        "prominences": prominences,
        "ref_prom": ref_prom,
        "period_px": period_px,
    }
    return nozzle_edge_y, float(period_px), ref_prom, debug

def detect_nozzle_x_range(img, nozzle_edge_y: int,
        center_frac: Tuple[float, float] = (0.35, 0.7),
        edge_margin_px: int = 0,
        thresh_frac: float = 2.5,
        debug: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Detect nozzle x-range by looking for columns with LOW fringe energy (horizontal fringes).
    Returns: x_left, x_right, x_center, debug
    """
    def threshold_calculation(profile: np.ndarray, threshold_frac) -> float:
        """
        Compute Otsu threshold for a 1D profile by quantizing to uint8 and using cv2.threshold.
        Works well when the profile has two populations: low (nozzle) and high (background).
        """
        p = profile.astype(np.float32)
        p = p[np.isfinite(p)]
        if p.size < 10:
            return float(np.nan)

        # 1. Calculate robust statistics
        p_min = np.min(profile)
        p_max = np.max(profile)
        p_mean = np.mean(profile)
        p_median = np.median(profile)

        # Robust scaling to [0,255] to reduce sensitivity to outliers
        lo = np.percentile(p, 1)
        hi = np.percentile(p, 99)
        pu8 = np.clip((p - lo) / (hi - lo), 0, 1)
        pu8 = (pu8 * 255).astype(np.uint8)

        thr_u8, _ = cv2.threshold(pu8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = lo + (thr_u8 / 255.0) * (hi - lo)
        return float(thr)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    h, w = img.shape
    x1 = int(w * center_frac[0])
    x2 = int(w * center_frac[1])

    band = img[int(nozzle_edge_y):h, :]
    band = band - band.mean(axis=0, keepdims=True)
    # normalize the band so the middle of the image is 0
    band = band / (np.std(band[:, x1 : x2 ], axis=1, keepdims=True) + 1e-6)
    band = np.abs(band)
    profile = band.mean(axis=0)
    profile -= profile.min()
    kernel = np.ones(60) / 60
    profile = np.convolve(profile, kernel, mode='same')

    # original simple thresholding method:
    threshold = np.mean(profile[x1 : x2]) * thresh_frac
    x_center = w // 2
    x_left = None
    x_right = None
    for offset in range(w // 2):
        if x_left is None:
            x_left_candidate = x_center - offset
            if profile[x_left_candidate] > threshold:
                x_left = x_left_candidate + edge_margin_px
        if x_right is None:
            x_right_candidate = x_center + offset
            if profile[x_right_candidate] > threshold:
                x_right = x_right_candidate - edge_margin_px
        if x_left is not None and x_right is not None:
            break

    if debug:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Panel 1: Image ROI with Edge & X-Range
        ax[0].imshow(img, cmap='gray', aspect='auto')
        ax[0].axhline(nozzle_edge_y, color='red', linestyle='--', linewidth=1.5,
                      label=f'Nozzle Edge (y={nozzle_edge_y})')
        if x_left is not None:
            ax[0].axvline(x_left, color='cyan', linestyle='--', linewidth=1.5, label=f'x_left ({x_left}px)')
        if x_right is not None:
            ax[0].axvline(x_right, color='magenta', linestyle='--', linewidth=1.5,
                          label=f'x_right ({x_right}px)')
        ax[0].set_title('ROI Image with Nozzle Edge and X Range Detection')
        ax[0].set_ylabel('y [px]')
        ax[0].legend(loc='upper right')

        # Panel 2: 1D Fringe Energy Profile & Threshold
        ax[1].plot(profile, color='blue', label='Fringe Energy Profile')
        ax[1].axhline(threshold, color='red', linestyle=':', label=f'Threshold ({threshold:.2f})')
        if x_left is not None:
            ax[1].axvline(x_left, color='cyan', linestyle='--')
        if x_right is not None:
            ax[1].axvline(x_right, color='magenta', linestyle='--')
        ax[1].set_xlabel('x [px]')
        ax[1].set_ylabel('Fringe Energy')
        ax[1].legend(loc='upper right')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return x_left, x_right



def calculate_px_to_mm(x_left: int, x_right: int, nozzle_width_mm: float) -> float:
    nozzle_width_px = x_right - x_left
    if nozzle_width_px <= 0:
        raise ValueError("x_right must be greater than x_left")

    px_to_mm = nozzle_width_mm / nozzle_width_px
    return px_to_mm


def calibrate_phase(unwrapped_phase: np.ndarray, nozzle_edge_y: int, nozzle_left_px: int, nozzle_right_px: int,
                   margin_px: int = 100, ref_y_offset: int = 750, debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    Zeros out the unwrapped phase map using a 1D polynomial fit (tilt removal).
    The fit is calculated using the background regions (left and right of the nozzle)
    from the nozzle edge up to a specified height (ref_y_offset).

    Args:
        unwrapped_phase (np.ndarray): The 2D unwrapped phase map.
        nozzle_edge_y (int): Y-coordinate of the nozzle edge.
        nozzle_left_px (int): X-coordinate of the left nozzle edge.
        nozzle_right_px (int): X-coordinate of the right nozzle edge.
        margin_px (int): Pixels away from the nozzle edge to exclude from the background.
        ref_y_offset (int): How many pixels above the nozzle edge to use for the background region.

    Returns:
        tuple:
            - np.ndarray: The calibrated (zeroed) phase map.
            - dict: Diagnostics including the fitted slope, intercept, and bounds used.
    """
    h, w = unwrapped_phase.shape

    # Define the vertical bounds for the background region
    # y=0 is the top of the image, so the nozzle edge is a higher index.
    y_start = max(0, int(nozzle_edge_y) - ref_y_offset)
    y_end = int(nozzle_edge_y)

    # Define the horizontal bounds for the background region
    left_bg_end = max(0, nozzle_left_px - margin_px)
    right_bg_start = min(w, nozzle_right_px + margin_px)

    # We want to fit a line to the background pixels to remove tilt across the x-axis.
    # We will average all the valid rows in the background region to get a stable profile.
    background_rows = unwrapped_phase[y_start:y_end, :]
    mean_profile = np.nanmean(background_rows, axis=0)

    # Create an array of x-coordinates (column indices)
    x = np.arange(w)

    # Select the x and y values corresponding to the background regions
    bg_mask = (x < left_bg_end) | (x > right_bg_start)

    x_bg = x[bg_mask]
    y_bg = mean_profile[bg_mask]

    # Remove any NaNs that might have resulted from the mean calculation
    valid = np.isfinite(y_bg)
    x_bg = x_bg[valid]
    y_bg = y_bg[valid]

    if len(x_bg) < 2:
        raise ValueError(
            "Not enough background pixels found to perform a fit. Adjust the margin or check edge detection.")

    # Fit a 1st degree polynomial (a line: y = mx + c) to the background
    coefficients = np.polyfit(x_bg, y_bg, 1)
    slope = coefficients[0]
    intercept = coefficients[1]

    # Create the background trend line for all x coordinates
    background_trend = (slope * x) + intercept

    # Subtract the 1D trend from every row in the 2D phase map
    # Broadcasting will automatically subtract this 1D array from every row in the 2D array
    calibrated_phase = unwrapped_phase - background_trend

    debug_info = {
        "slope": slope,
        "intercept": intercept,
        "left_bg_width": left_bg_end,
        "right_bg_width": w - right_bg_start,
        "y_start": y_start,
        "y_end": y_end
    }

    if debug:
        limit = np.nanmax(np.abs(calibrated_phase))
        norm = TwoSlopeNorm(
            vmin=-limit,
            vcenter=0,
            vmax=limit
        )
        plt.figure()
        im = plt.imshow(
            calibrated_phase,
            cmap="seismic",
            norm=norm,
            aspect="auto"
        )
        plt.colorbar(im, label = "Phase [rad]")
        plt.show()

    return calibrated_phase, debug_info