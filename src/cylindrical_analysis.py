
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Physical Constants & Gas Definitions
# =============================================================================

WAVELENGTH_M: float = 532e-9  # Probe laser wavelength [m]
AVOGADRO_NA: float = 6.022e23  # Avogadro constant [mol^-1]

GAS_PROPERTIES: Dict[str, Dict[str, float]] = {
    "argon": {
        "K": 1.57e-4,   # Gladstone-Dale constant [m^3/kg]
        "M": 39.95e-3,  # Molar mass [kg/mol]
    },
    "helium": {
        "K": 1.96e-4,   # Gladstone-Dale constant [m^3/kg]
        "M": 4.0026e-3, # Molar mass [kg/mol]
    },
}


# =============================================================================
# 1. Cropping & Spatial Formatting
# =============================================================================

def crop_and_format_phase_map(
    phase_map: np.ndarray,
    nozzle_edge_y: int,
    px_to_mm : float,
    max_height_mm: float = 6.0,
    nozzle_row: str = "last",
) -> np.ndarray:
    """
    Crop the phase map to the region of interest above the nozzle and orient it
    so that row index 0 corresponds to the physical nozzle exit.

    Parameters
    ----------
    phase_map : np.ndarray
        2D calibrated phase map array.
    nozzle_edge_y : int, optional
        Y-pixel index of the nozzle exit edge (e.g., from calib.estimate_nozzle_edge).
    px_to_mm : float
        Spatial scaling factor in mm per pixel.
    max_height_mm : float
        Maximum height above the nozzle to retain in mm (default: 6.0 mm).
    nozzle_row : str
        'last' if nozzle exit is at higher row index (jet flows upwards in image),
        'first' if nozzle exit is at lower row index (jet flows downwards).

    Returns
    -------
    np.ndarray
        Cropped 2D phase map oriented with row 0 at the nozzle exit edge.
    """
    arr = np.asarray(phase_map)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D phase map, got shape {arr.shape}")

    h, _ = arr.shape
    max_height_px = int(round(max_height_mm / px_to_mm)) if px_to_mm > 0 else h

    if nozzle_edge_y is not None:
        edge = int(nozzle_edge_y)
        if nozzle_row == "last":
            y_start = max(0, edge - max_height_px)
            cropped = arr[y_start:edge, :]
            return np.flipud(cropped)
        elif nozzle_row == "first":
            y_end = min(h, edge + max_height_px)
            return arr[edge:y_end, :]
        else:
            raise ValueError("nozzle_row must be 'first' or 'last'")
    else:
        proc = np.flipud(arr) if nozzle_row == "last" else arr
        return proc[:max_height_px, :]


# =============================================================================
# 2. Abel Inversion Core
# =============================================================================

def abel_inversion_1d(line: np.ndarray, px_to_mm: float) -> np.ndarray:
    """
    Perform 1D numerical Abel inversion on a radial phase profile.

    Assumes line[r] is the line-of-sight optical phase integral at radius r.
    Uses central differences in physical coordinates (mm) and numerical
    integration:
        f(r) = -1/pi * int_r^R (dF/ds) / sqrt(s^2 - r^2) ds

    Parameters
    ----------
    line : np.ndarray
        1D array of phase values from r=0 to outer radius R.
    px_to_mm : float
        Spatial resolution in mm per pixel.

    Returns
    -------
    np.ndarray
        Inverted radial profile in units of rad/mm.
    """
    n = len(line)
    inverted = np.zeros(n, dtype=np.float64)
    if n < 2 or px_to_mm <= 0:
        return inverted

    # Central difference in physical space (rad/mm)
    dF_ds = np.zeros(n, dtype=np.float64)
    dF_ds[0] = (line[1] - line[0]) / px_to_mm
    dF_ds[-1] = (line[-1] - line[-2]) / px_to_mm
    dF_ds[1:-1] = (line[2:] - line[:-2]) / (2.0 * px_to_mm)

    trapz_fn = getattr(np, "trapezoid", np.trapz)

    for r in range(n - 1):
        r_mm = float(r) * px_to_mm
        s_indices = np.arange(r + 1, n, dtype=np.int32)
        s_mm = s_indices.astype(np.float64) * px_to_mm

        s_squared = s_mm ** 2
        r_squared = r_mm ** 2
        denominators = np.sqrt(np.maximum(s_squared - r_squared, 1e-10))
        weights = 1.0 / denominators

        integrand = dF_ds[s_indices] * weights
        integral = trapz_fn(integrand, x=s_mm)
        inverted[r] = -integral / np.pi

    return inverted


def abel_invert_phase_map(
    phase_map: np.ndarray,
    px_to_mm: float,
    center_x: Optional[Union[int, float]] = None,
    unwrap_rows: bool = True,
    jump_thr: float = 1.6 * np.pi,
) -> np.ndarray:
    """
    Apply row-by-row Abel inversion across an axisymmetric 2D phase map.

    Symmetry strategy:
    1. Splits each horizontal row at center_x into left and right radial halves.
    2. Averages both halves to leverage symmetry and reduce experimental noise.
    3. Inverts the radial line via Abel transformation.
    4. Mirrors the inverted result across the center to reconstruct the 2D field.

    Parameters
    ----------
    phase_map : np.ndarray
        2D phase map array where axis 0 is height (y) and axis 1 is radial (x).
    px_to_mm : float
        Spatial resolution in mm per pixel.
    center_x : int or float, optional
        X-pixel index of the jet symmetry axis. Defaults to width // 2.
    unwrap_rows : bool
        Whether to check and unwrap residual 2pi phase jumps across rows.
    jump_thr : float
        Threshold for detecting phase wrap jumps in a row.

    Returns
    -------
    np.ndarray
        Abel-inverted 2D array of the same shape (units: rad/mm).
    """
    arr = np.asarray(phase_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    h, w = arr.shape
    center = int(round(center_x)) if center_x is not None else w // 2
    inverted_total = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        row = arr[i, :].copy()

        if unwrap_rows:
            d = np.diff(row)
            if np.any(np.abs(d) > float(jump_thr)):
                row = np.unwrap(row)

        left_radial = row[: center + 1][::-1]
        right_radial = row[center:]

        n_radial = min(len(left_radial), len(right_radial))
        if n_radial <= 0:
            continue

        left_radial = left_radial[:n_radial]
        right_radial = right_radial[:n_radial]

        radial_line = 0.5 * (left_radial + right_radial)

        if n_radial > 1:
            inv_rad = abel_inversion_1d(radial_line, px_to_mm=px_to_mm)
        else:
            inv_rad = radial_line.copy()

        # Reconstruct full row with mirror symmetry
        inverted_total[i, center - n_radial + 1 : center + 1] = inv_rad[::-1]
        inverted_total[i, center : center + n_radial] = inv_rad

    return inverted_total.astype(np.float32)


# =============================================================================
# 3. Density Conversion (Gladstone-Dale)
# =============================================================================

def phase_to_density(
    phase_map: np.ndarray,
    px_to_mm: float,
    gas: str = "argon",
    wavelength_m: float = WAVELENGTH_M,
    center_x: Optional[Union[int, float]] = None,
    is_inverted: bool = False,
    clip_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an axisymmetric phase map to number density (cm^-3).

    Performs Abel inversion (if not already performed) and converts radial
    refractivity to mass density via Gladstone-Dale, then to number density.

    Parameters
    ----------
    phase_map : np.ndarray
        2D phase map oriented with row 0 at the nozzle edge.
    px_to_mm : float
        Pixel scaling factor in mm/pixel.
    gas : str
        Gas species ('argon' or 'helium').
    wavelength_m : float
        Laser wavelength in meters (default: 532 nm).
    center_x : int or float, optional
        Horizontal center index for the symmetry axis.
    is_inverted : bool
        Set to True if phase_map has already undergone Abel inversion.
    clip_negative : bool
        Clip non-physical negative density values to 0.

    Returns
    -------
    density_map : np.ndarray
        2D array of number densities in cm^-3.
    x_mm : np.ndarray
        1D horizontal spatial axis in mm.
    y_mm : np.ndarray
        1D vertical spatial axis in mm (0 = nozzle exit edge).
    """
    gas_key = gas.lower()
    if gas_key not in GAS_PROPERTIES:
        raise ValueError(
            f"Unsupported gas: '{gas}'. Supported gases: {list(GAS_PROPERTIES.keys())}"
        )

    k_gd = GAS_PROPERTIES[gas_key]["K"]
    m_mol = GAS_PROPERTIES[gas_key]["M"]

    phase_arr = np.asarray(phase_map)
    h, w = phase_arr.shape

    if is_inverted:
        phase_inv = phase_arr.astype(np.float64)
    else:
        phase_inv = abel_invert_phase_map(
            phase_arr, px_to_mm=px_to_mm, center_x=center_x
        ).astype(np.float64)

    # Convert rad/mm -> rad/m
    phase_inv_per_m = phase_inv * 1e3

    # Gladstone-Dale: rho = (dPhi/dr) * lambda / (2 * pi * K_GD) [kg/m^3]
    rho = (phase_inv_per_m * wavelength_m) / (2.0 * np.pi * k_gd)

    # Number density: n = (rho / (M / N_A)) * 1e-6 [cm^-3]
    density_cm3 = (rho / (m_mol / AVOGADRO_NA)) / 1e6

    if clip_negative:
        density_cm3 = np.maximum(density_cm3, 0.0)

    y_mm = np.arange(h, dtype=np.float32) * px_to_mm
    x_mm = np.arange(w, dtype=np.float32) * px_to_mm

    return density_cm3, x_mm, y_mm


def extract_radial_profile(
    density_map: np.ndarray,
    y_mm: np.ndarray,
    x_mm: np.ndarray,
    y_target_mm: float,
) -> Tuple[np.ndarray, np.ndarray, float]:

    iy = int(np.argmin(np.abs(y_mm - float(y_target_mm))))
    return x_mm, density_map[iy, :], float(y_mm[iy])


# =============================================================================
# 4. Visualization Helpers
# =============================================================================

def plot_density_map(
    density_map: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    title: str = "Cylindrical Nozzle Density Map",
    robust_pct: float = 98.0,
    y_profile_mm: Optional[float] = None,
    cmap: str = "viridis",
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the 2D reconstructed gas density map with calibrated physical axes."""
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor="white", constrained_layout=True)
    ax.set_facecolor("white")

    vmax = float(np.nanpercentile(density_map, robust_pct))
    vmax = max(vmax, 1e-12)
    extent = [float(x_mm[0]), float(x_mm[-1]), float(y_mm[0]), float(y_mm[-1])]

    im = ax.imshow(
        density_map,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    if y_profile_mm is not None:
        ax.axhline(
            y=y_profile_mm,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Profile at y={y_profile_mm:.2f} mm",
        )
        ax.legend(loc="upper right")

    ax.set_title(title, color="black", fontsize=13)
    ax.set_xlabel("x (mm)", color="black", fontsize=11)
    ax.set_ylabel("Height above nozzle y (mm)", color="black", fontsize=11)
    ax.tick_params(colors="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number Density (cm$^{-3}$)", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

    if show:
        plt.show()

    return fig, ax


def plot_radial_profiles(
    density_map: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    y_targets_mm: List[float],
    title: str = "Radial Density Profiles",
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple 1D density profiles extracted at different heights."""
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white", constrained_layout=True)
    ax.set_facecolor("white")

    for y_tgt in y_targets_mm:
        _, prof, actual_y = extract_radial_profile(density_map, y_mm, x_mm, y_tgt)
        ax.plot(x_mm, prof, linewidth=1.8, label=f"y = {actual_y:.2f} mm")

    ax.set_title(title, color="black", fontsize=13)
    ax.set_xlabel("x (mm)", color="black", fontsize=11)
    ax.set_ylabel("Number Density (cm$^{-3}$)", color="black", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.tick_params(colors="black")

    if show:
        plt.show()

    return fig, ax


