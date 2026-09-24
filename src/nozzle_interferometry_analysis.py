from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt, use
from skimage.restoration import unwrap_phase
import os
from datetime import datetime
import json

from data_analysis import start_analysis
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any, Union
import numpy.typing as npt
import re
from pathlib import Path
from tkinter import Tk, filedialog



# type aliases
FileName = str
Pair = Tuple[FileName, FileName]
PhaseMap = npt.NDArray[np.float32]  # or np.floating if you want more general
Delay = float

@dataclass(frozen=True)
class PhaseMapMetadata:
    """Metadata for a generated phase map with physical dimensions and scaling info."""
    nozzle_width_mm: float  # physical width of the nozzle (reference: 6.0 mm)
    nozzle_x_left_px: int   # left edge of nozzle in raw image pixels
    nozzle_x_right_px: int  # right edge of nozzle in raw image pixels
    roi_width_mm: float     # actual width of the saved ROI in mm (including expansion)
    px_per_mm: float        # pixels per mm (for correct scaling)
    height_mm: float        # actual height of the saved ROI in mm
    expansion_mm: float     # expansion beyond nozzle edges (per side)

# Helper function for al classes
def ask_for_folder(*, title: str, initialdir: str | Path | None = None, mustexist: bool = True) -> Path:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(
        title=title,
        initialdir=str(initialdir) if initialdir is not None else None,
        mustexist=mustexist,
    )

    root.destroy()

    if not folder:
        raise RuntimeError("No folder selected (user cancelled).")

    p = Path(folder)
    if mustexist and not p.is_dir():
        raise FileNotFoundError(f"Selected folder does not exist: {p}")

    return p


class PhaseMapGenerator:

    def __init__(self, folder_path: str | Path | None = None):
        if folder_path is None or str(folder_path).strip() == "":
            folder_path = ask_for_folder(
                title="Select nozzle interferometry folder",
                initialdir= r"C:/Users/vmlab/Documents/data/files"
            )
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
        self.analysis = start_analysis(folder_path)

    def save_phase_png(self, delay: float, tol: float = 1e-6):
        """
        Calculate the phase map for a given delay and save the same figure
        produced by debug_phase() as a PNG.
        """

        # ---- find pair ----
        pairs = self.find_interferometry_pairs()
        match = None

        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue

        if match is None:
            raise FileNotFoundError(
                f"No pair found for delay={delay} (tol={tol})"
            )

        # ---- calculate phase ----
        phase, metadata = self.get_interferometry_phase(match)

        # ---- physical dimensions ----
        h_px, w_px = phase.shape
        px_per_mm = metadata.px_per_mm
        mm_per_px = 1.0 / px_per_mm

        height_mm = h_px * mm_per_px
        width_mm = metadata.roi_width_mm

        extent = [
            0,
            width_mm,
            0,
            height_mm
        ]

        # ---- create same figure as debug_phase ----
        fig, ax = plt.subplots()

        im = ax.imshow(
            phase,
            cmap="seismic",
            aspect="auto",
            extent=extent
        )

        ax.set_title(f"phase map — delay={delay:.6f}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        fig.colorbar(im, ax=ax, label="Phase (rad)")

        # ---- choose output folder ----
        out_dir = ask_for_folder(
            title="Select folder to save phase map"
        )

        delay_str = f"{float(delay):.6f}"
        phase_path = out_dir / f"delay_{delay_str}_phase.png"

        # ---- save figure ----
        fig.savefig(
            phase_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved phase figure: {phase_path}")

        return phase_path

    def save_interferometry_images_png(self, delay: float, tol: float = 1e-6):
        """
        Save cropped reference and experiment interferometry images as PNGs.
        The PNG intensity is normalized for visualization.
        """

        # ---- find pair ----
        pairs = self.find_interferometry_pairs()
        match = None

        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue

        if match is None:
            raise FileNotFoundError(
                f"No pair found for delay={delay} (tol={tol})"
            )

        # ---- load images ----
        ref_image, exp_image = self.get_interferometry_data(match)

        # ---- crop ----
        ref_image = ref_image[2000:-1, :]
        exp_image = exp_image[2000:-1, :]

        # ---- convert to uint16 for PNG ----
        def normalize_for_png(image):
            image = image.astype(np.float32)

            image_min = np.min(image)
            image_max = np.max(image)

            normalized = (
                    (image - image_min) /
                    (image_max - image_min)
                    * 65535
            )

            return normalized.astype(np.uint16)

        ref_png = normalize_for_png(ref_image)
        exp_png = normalize_for_png(exp_image)

        # ---- choose output folder ----
        out_dir = ask_for_folder(
            title="Select folder to save interferometry images"
        )

        delay_str = f"{float(delay):.6f}"

        ref_path = out_dir / f"delay_{delay_str}_ref.png"
        exp_path = out_dir / f"delay_{delay_str}_exp.png"

        # ---- save ----
        cv2.imwrite(str(ref_path), ref_png)
        cv2.imwrite(str(exp_path), exp_png)

        print(
            f"Reference image range: "
            f"{ref_image.min()} → {ref_image.max()}"
        )
        print(
            f"Experiment image range: "
            f"{exp_image.min()} → {exp_image.max()}"
        )

        print(f"Saved reference image: {ref_path}")
        print(f"Saved experiment image: {exp_path}")

        return ref_path, exp_path

    def debug_fourier(self, delay: float):
        # ---- find pair ----
        tol = 1e-6
        pairs = self.find_interferometry_pairs()
        match = None
        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue
        if match is None:
            raise FileNotFoundError(f"No pair found for delay={delay} (tol={tol})")
        ref_image, exp_image = self.get_interferometry_data(match)

        # find the area above the nozzle edge
        ref_edge, *_ = self.estimate_nozzle_edge(ref_image)
        exp_edge, *_ = self.estimate_nozzle_edge(exp_image)
        if ref_edge is None or exp_edge is None:
            raise ValueError("Nozzle edge detection failed")

        edge = max(ref_edge, exp_edge)

        x_left, x_right = self.detect_nozzle_x_range(ref_image, edge)
        if x_left is None or x_right is None:
            raise ValueError("Nozzle x-range detection failed")

        ref_roi = self.crop_image_roi(ref_image, edge, x_left, x_right)
        exp_roi = self.crop_image_roi(exp_image, edge, x_left, x_right)

        ref_filtered_fourier, _, _ = self.get_filtered_fourier(ref_roi, exp_roi, debug=True)

    def debug_phase(self, delay: float):
        # ---- find pair ----
        tol = 1e-6
        pairs = self.find_interferometry_pairs()
        match = None
        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue
        if match is None:
            raise FileNotFoundError(f"No pair found for delay={delay} (tol={tol})")
        phase, metadata = self.get_interferometry_phase(match)
        h_px, w_px = phase.shape
        px_per_mm = metadata.px_per_mm
        mm_per_px = 1.0 / px_per_mm
        height_mm = h_px * mm_per_px
        width_mm = metadata.roi_width_mm
        extent = [0, width_mm, 0, height_mm]  # x: 0 to width, y: 0 at bottom (nozzle edge)

        plt.figure()
        im = plt.imshow(phase, cmap="seismic", aspect="auto", extent=extent)
        plt.title(f"phase map — delay={delay:.6f}")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)  (0 = nozzle edge)")
        plt.colorbar(im, label="Phase (rad)")
        plt.show()

    def debug_nozzle_edge_and_xrange(self, delay: float):
        # ---- find pair ----
        tol = 1e-6
        pairs = self.find_interferometry_pairs()
        match = None
        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue
        if match is None:
            raise FileNotFoundError(f"No pair found for delay={delay} (tol={tol})")
        ref_image, exp_image = self.get_interferometry_data(match)

        ref_edge, *_ = self.estimate_nozzle_edge(ref_image)
        exp_edge, *_ = self.estimate_nozzle_edge(exp_image)
        if ref_edge is None or exp_edge is None:
            raise ValueError("Nozzle edge detection failed")

        edge = max(ref_edge, exp_edge)

        x_left, x_right = self.detect_nozzle_x_range(ref_image, edge, debug = True)
        if x_left is None or x_right is None:
            raise ValueError("Nozzle x-range detection failed")

        print(f"[delay={delay:.6f}] pair={match}")
        print(f"ref_edge={ref_edge} | exp_edge={exp_edge} | used edge={edge}")
        print(f"x_left={x_left} | x_right={x_right}")
    #     plot the edges and x-range on the reference image for visual confirmation
        plt.figure()
        plt.imshow(exp_image, cmap="gray", origin="upper")
        h, w = exp_image.shape
        plt.plot([0, w], [edge, edge], 'r--', label='Nozzle Edge')
        plt.plot([x_left, x_left], [0, h], 'g--', label='X Left')
        plt.plot([x_right, x_right], [0, h], 'b--', label='X Right')
        plt.title(f"Exp Image with Detected Edges — delay={delay:.6f}")
        plt.legend()
        plt.show()

        # out_dir = ask_for_folder(title='Select output folder for ROIs (for Neutrino)')
        # out_dir = Path(out_dir)
        # out_dir.mkdir(parents=True, exist_ok=True)
        # ref_path = out_dir / f"delay_{float(delay):.6f}_{match[0].split('.')[0]}_ref_roi.npy"
        # exp_path = out_dir / f"delay_{float(delay):.6f}_{match[1].split('.')[0]}_exp_roi.npy"
        # np.save(ref_path, ref_image)
        # np.save(exp_path, exp_image)

    def debug_plot_my_phase_vs_neutrino_tiff(self, delay: float, neutrino_tiff_path: str | Path, tol: float = 1e-6,
            *,show: bool = True,):
        """
        Minimal debug:
        - Compute dphi using get_interferometry_phase (production)
        - Load Neutrino TIFF
        - Plot side-by-side (no alignment, no scaling tricks)

        Assumes the Neutrino TIFF was produced from debug_oscillation_characterization_v2
        (i.e., it is a uint16-scaled image of the same data/ROI).
        """

        import numpy as np
        import cv2
        from pathlib import Path
        import matplotlib.pyplot as plt

        # ---- find pair for delay ----
        pairs = self.find_interferometry_pairs()
        match = None
        for p in pairs:
            try:
                d = float(self.get_hdf5_experiment_image_trigger_delay(p[0]))
                if abs(d - float(delay)) <= float(tol):
                    match = p
                    break
            except Exception:
                continue
        if match is None:
            raise FileNotFoundError(f"No pair found for delay={delay} (tol={tol})")

        # ---- my phase (float radians) ----
        dphi = self.get_interferometry_phase(match).astype(np.float32)

        # ---- load Neutrino TIFF (most likely uint16) ----
        neutrino_tiff_path = Path(neutrino_tiff_path)

        img = cv2.imread(str(neutrino_tiff_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read: {neutrino_tiff_path}")
        if img.ndim == 3:
            img = img.mean(axis=2)

        neu = img.astype(np.float32)
        dphi = dphi / (2 * np.pi)
        # ---- print stats ----
        print(f"[delay={delay:.6f}] pair={match}")
        print(
            f"my dphi:   shape={dphi.shape}  min={float(dphi.min()):+.3f}  max={float(dphi.max()):+.3f}  mean={float(dphi.mean()):+.3f}  std={float(dphi.std()):.3f}")
        print(
            f"neutrino:  shape={neu.shape}   min={float(neu.min()):+.1f}  max={float(neu.max()):+.1f}  mean={float(neu.mean()):+.1f}  std={float(neu.std()):.1f}")

        if not show:
            return {"pair": match, "dphi": dphi, "neutrino": neu}

        dphi = dphi[0:neu.shape[0], 0:neu.shape[1]]
        W_mm = 6.0
        w_px = dphi.shape[1]
        mm_per_px = W_mm / w_px
        x_mm = np.arange(w_px) * mm_per_px

        plt.figure(figsize=(10,5))
        mm_to_check = [2,3.5,5]
        for mm in mm_to_check:
            px = int(mm / mm_per_px)
            dphi_prof = dphi[px,:]
            neu_prof = neu[px, :]
            plt.plot(x_mm, dphi_prof, label=f'calculated phase at y={mm} mm')
            plt.plot(x_mm, neu_prof, label=f'neutrion phase at y={mm} mm', linestyle='dashed')
        plt.title(f"Lineout comparison at delay={delay:.6f}")
        plt.xlabel("x (mm)")
        plt.ylabel("Phase")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        # return {"pair": match, "dphi": dphi, "neutrino": neu}

    def _file_path(self, file_name) -> str:
        return str(Path(self.analysis.folder_path) / file_name)

    def get_hdf5_time_data(self, file_name: FileName) -> Any:
        file_path = self._file_path(file_name)
        time_data = self.analysis.get_hdf5_group_data(file_path, 'main')['time']
        return time_data

    def get_hdf5_parker_valve_state(self, file_name: FileName) -> Any:
        file_path = self._file_path(file_name)
        parker_state = self.analysis.get_hdf5_group_data(file_path, 'pulse_generators/qc_a/pulse_generator_channels/parker_valve_trigger')['pulse_state']
        return parker_state

    def get_hdf5_experiment_image_trigger_delay(self, file_name: FileName) -> Any:
        file_path = self._file_path(file_name)
        delay = self.analysis.get_hdf5_group_data(file_path, 'pulse_generators/qc_a/pulse_generator_channels/experiment_image_trigger')['delay']
        return delay

    def find_interferometry_pairs(self) -> List[Pair]:
        parameter_name = 'delay'
        parameter_dict: Dict[str, Dict[str, Any]] = self.analysis.create_hdf5_dict(
            "pulse_generators/qc_a/pulse_generator_channels/experiment_image_trigger"
        )

        waiting: Dict[Any, FileName] = {}
        pairs: List[Pair] = []

        for file_name, parameters in parameter_dict.items():
            parameter_value = parameters[parameter_name]
            if parameter_value in waiting:
                found_pair = waiting.pop(parameter_value)

                file_time = self.get_hdf5_time_data(file_name)
                file_valve_state = self.get_hdf5_parker_valve_state(file_name)

                found_time = self.get_hdf5_time_data(found_pair)
                found_valve_state = self.get_hdf5_parker_valve_state(found_pair)

                if file_time < found_time and file_valve_state == "off" and found_valve_state == "on":
                    pairs.append((file_name, found_pair))
                elif found_time < file_time and found_valve_state == "off" and file_valve_state == "on":
                    pairs.append((found_pair, file_name))
                else:
                    print(f"Warning: Could not match valve states for files {file_name} and {found_pair}")
                    print(f"File 1: {file_name}, Time start: {file_time}, Valve state: {file_valve_state}")
                    print(f"File 2: {found_pair}, Time start: {found_time}, Valve state: {found_valve_state}")
            else:
                waiting[parameter_value] = file_name

        if waiting:
            print(f"Warning: Unmatched scans found for parameter values: {list(waiting.keys())}")

        return pairs

    def get_interferometry_data(self, pair: Pair) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        file_1_path = self._file_path(pair[0])
        file_2_path = self._file_path(pair[1])
        data_1 = self.analysis.get_image_from_file(file_1_path)
        data_2 = self.analysis.get_image_from_file(file_2_path)
        return data_1, data_2

    def get_interferometry_phase(self, file_pair, expansion_mm: float = 3.0, nozzle_width_mm: float = 6.0):
        """
        Compute the wrapped phase difference map for one reference/experiment pair.

        Pipeline:
        - Detect nozzle edge on both images
        - Detect nozzle x-range
        - Calculate expanded ROI (nozzle width + expansion on each side)
        - Crop ROI above nozzle with expansion
        - Fourier filter around dominant fringe frequency
        - Demodulate
        - Unwrap ref/exp phases and compute delta
        - Wrap delta to (-pi, pi) and baseline-subtract

        Args:
            file_pair: (ref_file, exp_file)
            expansion_mm: additional width to capture beyond nozzle edges (per side), default 3.0 mm
            nozzle_width_mm: reference nozzle width in mm (for scaling), default 6.0 mm

        Returns:
            Tuple of (phase_difference, metadata)
            - phase_difference: 2D float32 array: delta phase wrapped to (-pi, pi)
            - metadata: PhaseMapMetadata with scaling information
        """
        ref_image, exp_image = self.get_interferometry_data(file_pair)

        # find the area above the nozzle edge
        ref_edge, *_ = self.estimate_nozzle_edge(ref_image)
        exp_edge, *_ = self.estimate_nozzle_edge(exp_image)
        if ref_edge is None or exp_edge is None:
            raise ValueError("Nozzle edge detection failed")

        edge = max(ref_edge, exp_edge)

        x_left, x_right = self.detect_nozzle_x_range(ref_image, edge)
        if x_left is None or x_right is None:
            raise ValueError("Nozzle x-range detection failed")

        # ---- Calculate nozzle metrics and expanded ROI bounds ----
        nozzle_width_px = x_right - x_left
        px_per_mm = nozzle_width_px / float(nozzle_width_mm)
        expansion_px = int(round(expansion_mm * px_per_mm))

        # Expanded bounds (clamped to image dimensions)
        h, w = ref_image.shape
        x_left_expanded = max(0, x_left - expansion_px)
        x_right_expanded = min(w, x_right + expansion_px)

        final_roi_width_px = x_right_expanded - x_left_expanded
        final_roi_width_mm = final_roi_width_px / px_per_mm

        # ---- Crop with expanded bounds ----
        ref_roi = self.crop_image_roi(ref_image, edge, x_left_expanded, x_right_expanded)
        exp_roi = self.crop_image_roi(exp_image, edge, x_left_expanded, x_right_expanded)

        final_height_px = ref_roi.shape[0]
        final_height_mm = final_height_px / px_per_mm

        ref_filtered_fourier, exp_filtered_fourier, dominant_frequency = self.get_filtered_fourier(ref_roi, exp_roi)
        ref_demodulated = self.demodulate_image(ref_filtered_fourier, dominant_frequency)
        exp_demodulated = self.demodulate_image(exp_filtered_fourier, dominant_frequency)

        phase_difference = self.extract_phase_difference(ref_demodulated, exp_demodulated)

        bg_mask = self.make_bg_mask_edge_boxes_physical(phase_difference.shape, roi_width_mm=final_roi_width_mm)
        # apply mask to baseline region only (not the whole image) to avoid edge artifacts dominating the baseline
        baseline_mask = bg_mask
        if np.any(baseline_mask):
            baseline_value = np.median(phase_difference[baseline_mask])
            phase_difference -= baseline_value

        # ---- Remove edge artifacts by cropping 0.2 mm from both sides ----
        crop_mm = 0.2
        crop_px = int(round(crop_mm * px_per_mm))
        if crop_px > 0 and phase_difference.shape[1] > 2 * crop_px:
            phase_difference = phase_difference[:, crop_px:-crop_px]
            # Update roi_width_mm to reflect the cropped width
            final_roi_width_mm = phase_difference.shape[1] / px_per_mm

        # ---- Create metadata ----
        metadata = PhaseMapMetadata(
            nozzle_width_mm=float(nozzle_width_mm),
            nozzle_x_left_px=int(x_left),
            nozzle_x_right_px=int(x_right),
            roi_width_mm=float(final_roi_width_mm),
            px_per_mm=float(px_per_mm),
            height_mm=float(final_height_mm),
            expansion_mm=float(expansion_mm),
        )

        return phase_difference, metadata

    def export_phase_maps_npy_by_delay(self, out_root=r"C:\Users\vmlab\Documents\data\phase_maps", interferometry_name=None, overwrite_existing=False):
        """
        Export phase maps for all interferometry pairs as .npy files.
        Folder structure:
            out_root / year / month / day / interferometry_name /
                    delay_<value>.npy
        Returns:
            out_dir (str),
            saved (list of dicts),
            failed (list of dicts)
        """

        def _safe_name(s: str) -> str:
            # Windows-safe filename
            bad = '<>:"/\\|?*'
            for ch in bad:
                s = s.replace(ch, "_")
            return s.strip().strip(".")

        def _extract_ymd_from_path(p: str) -> Tuple[str, str, str]:
            """
            Extract YYYY/MM/DD from a path like ...\\2026\\01\\14\\scan_name
            """
            parts = Path(p).parts
            for i in range(len(parts) - 2):
                if (
                        re.fullmatch(r"\d{4}", parts[i]) and
                        re.fullmatch(r"\d{2}", parts[i + 1]) and
                        re.fullmatch(r"\d{2}", parts[i + 2])
                ):
                    return parts[i], parts[i + 1], parts[i + 2]
            raise ValueError(f"Could not find YYYY\\MM\\DD in path: {p}")

        def _delay_to_str(delay_val) -> str:
            try:
                if isinstance(delay_val, (list, tuple, np.ndarray)):
                    if np.size(delay_val) == 1:
                        delay_val = float(np.ravel(delay_val)[0])
                    else:
                        return _safe_name(str(delay_val))
                else:
                    delay_val = float(delay_val)
                return f"{delay_val:.6f}"
            except Exception:
                return _safe_name(str(delay_val))

        # Infer interferometry name from scan folder if not given
        if interferometry_name is None:
            interferometry_name = os.path.basename(
                os.path.normpath(self.analysis.folder_path)
            )
        interferometry_name = _safe_name(interferometry_name)

        year, month, day = _extract_ymd_from_path(self.analysis.folder_path)
        out_dir = os.path.join(out_root, year, month, day, interferometry_name)
        # ---------------------------------------------------------
        # Check if phase maps already exist
        # ---------------------------------------------------------
        if os.path.isdir(out_dir):
            existing_npy = [
                f for f in os.listdir(out_dir)
                if f.lower().endswith(".npy")
            ]

            if existing_npy and not overwrite_existing:
                print(
                    f"[export_phase_maps_npy_by_delay] "
                    f"Found {len(existing_npy)} existing .npy files in:\n  {out_dir}\n"
                    f"Skipping export (overwrite_existing=False)."
                )
                return out_dir, [], [
                    {
                        "reason": "existing phase maps found",
                        "n_files": len(existing_npy),
                        "folder": out_dir,
                    }
                ]
        if os.path.isdir(out_dir) and overwrite_existing:
            for f in os.listdir(out_dir):
                if f.lower().endswith(".npy"):
                    os.remove(os.path.join(out_dir, f))

        os.makedirs(out_dir, exist_ok=True)

        pairs = self.find_interferometry_pairs()

        saved = []
        failed = []

        # handle repeated delays
        delay_counts = {}

        for pair in pairs:
            try:
                # 1) compute phase and metadata
                phase_difference, metadata = self.get_interferometry_phase(pair)
                if phase_difference is None:
                    failed.append({
                        "pair": pair,
                        "reason": "phase_difference is None"
                    })
                    continue

                # 2) extract delay (use reference file)
                delay_val = self.get_hdf5_experiment_image_trigger_delay(pair[0])
                delay_str = _delay_to_str(delay_val)

                # 3) disambiguate identical delays
                delay_counts.setdefault(delay_str, 0)
                delay_counts[delay_str] += 1
                suffix = "" if delay_counts[delay_str] == 1 else f"_{delay_counts[delay_str]}"

                filename = f"delay_{delay_str}{suffix}.npy"
                file_path = os.path.join(out_dir, filename)

                # 4) save raw phase map (authoritative data)
                np.save(file_path, phase_difference.astype(np.float32))

                # 5) save metadata as JSON
                metadata_path = os.path.join(out_dir, filename.replace(".npy", ".json"))
                metadata_dict = {
                    "nozzle_width_mm": float(metadata.nozzle_width_mm),
                    "nozzle_x_left_px": int(metadata.nozzle_x_left_px),
                    "nozzle_x_right_px": int(metadata.nozzle_x_right_px),
                    "roi_width_mm": float(metadata.roi_width_mm),
                    "px_per_mm": float(metadata.px_per_mm),
                    "height_mm": float(metadata.height_mm),
                    "expansion_mm": float(metadata.expansion_mm),
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)

                saved.append({
                    "pair": pair,
                    "delay": delay_val,
                    "file_path": file_path,
                    "shape": phase_difference.shape,
                    "metadata_path": metadata_path,
                })

            except Exception as e:
                failed.append({
                    "pair": pair,
                    "reason": str(e),
                })

        print(f"[export_phase_maps_npy_by_delay] Output folder: {out_dir}")
        print(f"[export_phase_maps_npy_by_delay] Saved: {len(saved)} | Failed: {len(failed)}")

        return out_dir, saved, failed

    def export_phase_map_npy_for_delay(self, delay, out_root=r"C:\Users\vmlab\Documents\data\phase_maps", interferometry_name=None, overwrite_existing=False):
        """
        Export phase map(s) for a single delay as .npy file(s).
        Folder structure:
            out_root / year / month / day / interferometry_name /
                    delay_<value>.npy
        Returns:
            out_dir (str),
            saved (list of dicts),
            failed (list of dicts)
        """

        def _safe_name(s: str) -> str:
            bad = '<>:"/\\|?*'
            for ch in bad:
                s = s.replace(ch, "_")
            return s.strip().strip(".")

        def _extract_ymd_from_path(p: str):
            parts = Path(p).parts
            for i in range(len(parts) - 2):
                if (
                        re.fullmatch(r"\d{4}", parts[i]) and
                        re.fullmatch(r"\d{2}", parts[i + 1]) and
                        re.fullmatch(r"\d{2}", parts[i + 2])
                ):
                    return parts[i], parts[i + 1], parts[i + 2]
            raise ValueError(f"Could not find YYYY\\MM\\DD in path: {p}")

        def _delay_to_str(delay_val):
            try:
                if isinstance(delay_val, (list, tuple, np.ndarray)):
                    if np.size(delay_val) == 1:
                        delay_val = float(np.ravel(delay_val)[0])
                    else:
                        return _safe_name(str(delay_val))
                else:
                    delay_val = float(delay_val)
                return f"{delay_val:.6f}"
            except Exception:
                return _safe_name(str(delay_val))

        if interferometry_name is None:
            interferometry_name = os.path.basename(
                os.path.normpath(self.analysis.folder_path)
            )
        interferometry_name = _safe_name(interferometry_name)

        year, month, day = _extract_ymd_from_path(self.analysis.folder_path)
        out_dir = os.path.join(out_root, year, month, day, interferometry_name)
        os.makedirs(out_dir, exist_ok=True)

        tol = 1e-6
        pairs = self.find_interferometry_pairs()
        saved = []
        failed = []
        delay_counts = {}

        for pair in pairs:
            try:
                delay_val = float(self.get_hdf5_experiment_image_trigger_delay(pair[0]))
                # Compare with requested delay (allowing for float precision)
                if not abs(delay_val - float(delay)) < float(tol):
                    continue
                phase_difference, metadata = self.get_interferometry_phase(pair)
                if phase_difference is None:
                    failed.append({
                        "pair": pair,
                        "reason": "phase_difference is None"
                    })
                    continue
                delay_str = _delay_to_str(delay_val)
                delay_counts.setdefault(delay_str, 0)
                delay_counts[delay_str] += 1
                suffix = "" if delay_counts[delay_str] == 1 else f"_{delay_counts[delay_str]}"
                filename = f"delay_{delay_str}{suffix}.npy"
                file_path = os.path.join(out_dir, filename)
                np.save(file_path, phase_difference.astype(np.float32))

                # save metadata as JSON
                metadata_path = os.path.join(out_dir, filename.replace(".npy", ".json"))
                metadata_dict = {
                    "nozzle_width_mm": float(metadata.nozzle_width_mm),
                    "nozzle_x_left_px": int(metadata.nozzle_x_left_px),
                    "nozzle_x_right_px": int(metadata.nozzle_x_right_px),
                    "roi_width_mm": float(metadata.roi_width_mm),
                    "px_per_mm": float(metadata.px_per_mm),
                    "height_mm": float(metadata.height_mm),
                    "expansion_mm": float(metadata.expansion_mm),
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)

                saved.append({
                    "pair": pair,
                    "delay": delay_val,
                    "file_path": file_path,
                    "shape": phase_difference.shape,
                    "metadata_path": metadata_path,
                })
            except Exception as e:
                failed.append({
                    "pair": pair,
                    "reason": str(e),
                })

        print(f"[export_phase_map_npy_for_delay] Output folder: {out_dir}")
        print(f"[export_phase_map_npy_for_delay] Saved: {len(saved)} | Failed: {len(failed)}")
        return out_dir, saved, failed

    def make_bg_mask_edge_boxes_physical(self, shape: tuple[int, int],
            *, roi_width_mm: float = 6.1, side_width_mm: float = 1.0, base_total_mm: float = 1.0, drop_bottom_mm: float = 0.2,
    ) -> np.ndarray:
        """
        BG mask defined in *physical* units:
          - side boxes width = side_width_mm (at left/right edges inside ROI)
          - y band near nozzle edge: from drop_bottom_mm up to base_total_mm
            (i.e. height = base_total_mm - drop_bottom_mm), avoiding very bottom artifacts

        Uses:
          - roi_width_mm (known)
          - pixel_size_um (known) to convert y-mm to pixels
        """

        h, w = shape

        # conversion from known ROI width
        px_per_mm = w / roi_width_mm
        mm_per_px = roi_width_mm / w

        side_w_px = int(round(side_width_mm * px_per_mm))
        base_total_px = int(round(base_total_mm * px_per_mm))
        drop_px = int(round(drop_bottom_mm * px_per_mm))

        side_w_px = max(1, min(side_w_px, w // 2))
        base_total_px = max(1, min(base_total_px, h))
        drop_px = max(0, min(drop_px, base_total_px - 1))

        # nozzle edge is at bottom of ROI array (because crop is above nozzle edge)
        y1 = h - drop_px
        y0 = max(0, y1 - (base_total_px - drop_px))

        mask = np.zeros(shape, dtype=bool)
        mask[y0:y1, 0:side_w_px] = True
        mask[y0:y1, w - side_w_px:w] = True
        return mask

    @staticmethod
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

        plt.figure()
        plt.plot(p)

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

    @staticmethod
    def detect_nozzle_x_range(img: npt.NDArray[Any], nozzle_edge_y: int,
            center_frac: Tuple[float, float] = (0.35, 0.7),
            edge_margin_px: int = 40,
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

            # # 2. Distinguish Wide vs Narrow
            # # If median is in the bottom 30% of the range, it's likely a wide nozzle
            # if (p_median - p_min) < 0.3 * (p_max - p_min):
            #     # WIDE NOZZLE:
            #     print('wide nozzle detected based on median position; using mean-based threshold')
            #     return p_mean * 2

            # Robust scaling to [0,255] to reduce sensitivity to outliers
            lo = np.percentile(p, 1)
            hi = np.percentile(p, 99)
            pu8 = np.clip((p - lo) / (hi - lo), 0, 1)
            pu8 = (pu8 * 255).astype(np.uint8)

            thr_u8, _ = cv2.threshold(pu8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = lo + (thr_u8 / 255.0) * (hi - lo)
            return float(thr)

        def modified_otsu_threshold(profile: np.ndarray, thresh_frac = 1.3) -> float:
            p = profile.astype(np.float32)
            p = p[np.isfinite(p)]
            if p.size < 10:
                return float(np.nan)

            # 1. Calculate robust statistics
            p_min = np.min(profile)
            p_max = np.max(profile)
            p_mean = np.mean(profile)
            p_median = np.median(profile)
            lo = np.percentile(p, 1)
            hi = np.percentile(p, 99)
            pu8 = np.clip((p - lo) / (hi - lo), 0, 1)
            pu8 = (pu8 * 255).astype(np.uint8)

            hist = cv2.calcHist([pu8], [0], None, [256], [0, 256]).ravel()
            hist_norm = hist / hist.sum()  # Probabilities

            # 2. Variables for Otsu's math
            bins = np.arange(256)
            between_class_variances = np.zeros(256)

            for t in range(1, 256):
                # Split probabilities at threshold t
                w0 = np.sum(hist_norm[:t])  # Weight Background
                w1 = np.sum(hist_norm[t:])  # Weight Foreground

                if w0 == 0 or w1 == 0: continue

                # Calculate means
                u0 = np.sum(bins[:t] * hist_norm[:t]) / w0
                u1 = np.sum(bins[t:] * hist_norm[t:]) / w1

                # Calculate Between-Class Variance
                between_class_variances[t] = w0 * w1 * (u0 - u1) ** 2

            # 3. Find the peak (this is what cv2.THRESH_OTSU returns)
            otsu_thresh = np.argmax(between_class_variances)
            smooth_variance = cv2.GaussianBlur(between_class_variances, (1, 15), 0).flatten()
            first_deriv = np.diff(smooth_variance)
            max_deriv = np.max(first_deriv)
            plateau_start = np.where(first_deriv[:otsu_thresh] > 0.2 * max_deriv)[0][-1]

            second_deriv = np.diff(first_deriv)
            knee_index = np.argmin(second_deriv[:80])

            # 4. Plot it to "see"
            # plt.figure(figsize=(10, 5))
            # plt.plot(between_class_variances, color='red', label='Between-Class Variance')
            # plt.bar(bins, hist_norm * (between_class_variances.max() / hist_norm.max()), alpha=0.3,
            #         label='Histogram (Scaled)')
            # plt.axvline(otsu_thresh, color='blue', linestyle='--', label=f'Otsu Threshold: {otsu_thresh}')
            # plt.axvline(knee_index, color='green', linestyle='--', label=f'Knee Point: {knee_index}')
            # plt.axvline(plateau_start, color='purple', linestyle='--', label=f'Plateau Start: {plateau_start}')
            # plt.legend()
            # plt.title("Visualizing Otsu's Internal Decision")
            # plt.show()
            return float(lo + (plateau_start / 255.0) * (hi - lo))

        def windowed_rise(p: np.ndarray, w: int) -> np.ndarray:
            p = p.astype(np.float32)
            # pad so indices are valid
            ppad = np.pad(p, (w, w), mode="edge")
            # centered windowed difference
            return ppad[2 * w:] - ppad[:-2 * w] # length = len(p)

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

        # use otsu thresholding on the profile to find the nozzle region
        # threshold = threshold_calculation(profile[2:-2], thresh_frac)
        # threshold = modified_otsu_threshold(profile)
        # if debug:
            # plt.figure()
            # plt.plot(profile[2:-2], label='fringe energy profile (smoothed)')
            # plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.2f}')
            # plt.title('Fringe Energy Profile (mean over y)')
            # plt.xlabel('x (pixels)')
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

        # x_center = w // 2
        # x_left = None
        # x_right = None
        # for offset in range(w // 2):
        #     if x_left is None:
        #         x_left_candidate = x_center - offset
        #         if profile[x_left_candidate] > threshold:
        #             x_left = x_left_candidate
        #     if x_right is None:
        #         x_right_candidate = x_center + offset
        #         if profile[x_right_candidate] > threshold:
        #             x_right = x_right_candidate
        #     if x_left is not None and x_right is not None:
        #         break
        #
        # return x_left, x_right

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

        return x_left, x_right

    @staticmethod
    def crop_above_nozzle_fullwidth(img: np.ndarray, edge_y: int, height_px: int = 3000) -> np.ndarray:
        y1 = max(0, edge_y - height_px)
        y2 = max(0, edge_y)
        return img[y1:y2, :]  # full width

    @staticmethod
    def apply_circular_mask(F_shifted: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
        rows, cols = F_shifted.shape
        cy, cx = center
        y, x = np.ogrid[:rows, :cols]
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
        return F_shifted * mask.astype(np.float32)

    @staticmethod
    def circular_mean_phase(phi_wrapped: np.ndarray) -> float:
        return float(np.angle(np.mean(np.exp(1j * phi_wrapped))))

    @staticmethod
    def crop_image_roi(img: npt.NDArray[Any], y0: int, x_left: int, x_right: int) -> npt.NDArray[Any]:
        y1 = max(0, y0 - 4000)
        y2 = max(0, y0)
        return img[y1:y2, x_left:x_right]

    @staticmethod
    def get_filtered_fourier(ref_roi, exp_roi, radius=30, debug=False):
        def create_mask(shape, center, radius):
            mask = np.zeros(shape, dtype=np.uint8)
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask_area = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
            mask[mask_area] = 1
            return mask

        def create_gaussian_cut_mask(shape, center, radius):
            """
            Gaussian mask centered at `center`:
              - value = 1 at center
              - value = 0.01 at r = radius
              - hard cutoff (0) for r > radius
            """
            rows, cols = shape
            cy, cx = center
            y, x = np.ogrid[:rows, :cols]

            r2 = (x - cx) ** 2 + (y - cy) ** 2
            r = np.sqrt(r2)

            # sigma such that Gaussian(radius) = 0.01
            sigma = radius / np.sqrt(2.0 * np.log(100.0))

            g = np.exp(-r2 / (2.0 * sigma ** 2))
            g[r > radius] = 0.0

            return g.astype(np.float32)

        # move to fourier space
        ref_fourier = np.fft.fft2(ref_roi - np.mean(ref_roi))
        exp_fourier = np.fft.fft2(exp_roi - np.mean(exp_roi))
        ref_fourier_shifted = np.fft.fftshift(ref_fourier)
        exp_fourier_shifted = np.fft.fftshift(exp_fourier)

        # find dominant frequency
        magnitude_spectrum = np.abs(ref_fourier_shifted)
        flat_indices = np.argpartition(magnitude_spectrum.flatten(), -2)[-2:]
        peak_indices = np.array(np.unravel_index(flat_indices, magnitude_spectrum.shape)).T
        dominant_frequency = peak_indices[np.argmin(peak_indices[:, 0])]

        if debug:
        # plot the magnitude spectrum with the dominant frequency marked
            plt.figure(figsize=(6, 6))
            plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
            plt.scatter(dominant_frequency[1], dominant_frequency[0], color='red', marker='x')
            plt.title('Magnitude Spectrum with Dominant Frequency')
            plt.xlabel('Frequency Component (u)')
            plt.ylabel('Frequency Component (v)')
            plt.show()
            print("dominant_frequency (v, u):", dominant_frequency)

        # mask around dominant frequency
        # dominant_frequency_mask = create_mask(ref_roi.shape, dominant_frequency, radius)
        dominant_frequency_mask = create_gaussian_cut_mask(ref_roi.shape, dominant_frequency, radius)

        ref_filtered_fourier = ref_fourier_shifted * dominant_frequency_mask
        exp_filtered_fourier = exp_fourier_shifted * dominant_frequency_mask

        return ref_filtered_fourier, exp_filtered_fourier, dominant_frequency

    @staticmethod
    def demodulate_image(filtered_fourier, dominant_frequency):
        # shift the dominant frequency to the center
        rows, cols = filtered_fourier.shape
        crow, ccol = rows // 2, cols // 2
        shift_y = crow - dominant_frequency[0]
        shift_x = ccol - dominant_frequency[1]
        shifted_fourier = np.roll(np.roll(filtered_fourier, shift_y, axis=0), shift_x, axis=1)
        demodulated_image = np.fft.ifft2(shifted_fourier)
        return demodulated_image

    @staticmethod
    def extract_phase_difference(ref_demodulated, exp_demodulated):
        ref_phase = np.angle(ref_demodulated)
        ref_phase = unwrap_phase(ref_phase)
        exp_phase = np.angle(exp_demodulated)
        exp_phase = unwrap_phase(exp_phase)
        delta_phase = ref_phase - exp_phase

        delta_phase_wrapped = np.angle(np.exp(1j * delta_phase))
        return unwrap_phase(delta_phase_wrapped)


@dataclass(frozen=True)
class PhaseMapEntry:
    """One saved phase map on disk with its parsed delay and file path."""
    delay: Delay
    path: Path

class SlitNozzleAnalysis:
    """
    Analysis helper for slit-nozzle interferometry where phase maps were exported as .npy files.

    Expected filename format (from your generator):
        delay_<value>.npy
        delay_<value>_<occurrence>.npy

    Two folders:
        - perpendicular_dir: slit perpendicular to laser propagation
        - parallel_dir:      slit parallel to laser propagation

    Core feature:
        Pair maps across the two folders by matching delay values.
    """

    _DELAY_RE = re.compile(
        r"^delay_(?P<delay>[+-]?\d+(?:\.\d+)?)\.npy$",
        re.IGNORECASE,
    )

    def __init__(self,
        perpendicular_dir: str | Path | None = None, parallel_dir: str | Path | None = None,
        tol: float = 1e-6,):
        if perpendicular_dir is None or str(perpendicular_dir).strip() == "":
            pass
        if parallel_dir is None or str(parallel_dir).strip() == "":
            perpendicular_dir = ask_for_folder(
                title="Select PERPENDICULAR phase-map folder",
                initialdir= r"C:/Users/vmlab/Documents/data/phase_maps"
            )
            parallel_dir = ask_for_folder(
                title="Select PARALLEL phase-map folder",
                initialdir= r"C:/Users/vmlab/Documents/data/phase_maps"
            )

        self.perpendicular_dir = Path(perpendicular_dir)
        self.parallel_dir = Path(parallel_dir)
        self.tol = float(tol)

    @staticmethod
    def load_phase_map(path: Union[str, Path]) -> PhaseMap:
        path = Path(path)
        arr = np.load(str(path))
        return np.asarray(arr)

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> Optional[Dict[str, float]]:
        """Load metadata JSON for a phase map. Returns None if file doesn't exist."""
        path = Path(path)
        metadata_path = path.with_suffix('.json')
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def load_phase_map_with_metadata(self, path: Union[str, Path]) -> Tuple[PhaseMap, Optional[Dict[str, float]]]:
        """Load phase map and try to load accompanying metadata JSON."""
        phase_map = self.load_phase_map(path)
        metadata = self.load_metadata(path)
        return phase_map, metadata

    def load_phase_map_by_delay(self, folder: Union[str, Path], delay: float,
        *, tol: Optional[float] = None) -> Tuple[PhaseMapEntry, PhaseMap]:

        folder = Path(folder)
        entries = self.list_phase_maps(folder)

        # exact match first
        for e in entries:
            if e.delay == float(delay):
                return e, self.load_phase_map(e.path)

        # tolerant match
        tol_val = self.tol if tol is None else float(tol)
        if tol_val <= 0.0:
            raise FileNotFoundError(f"No exact delay={delay} in folder: {folder}")

        delays = np.array([e.delay for e in entries], dtype=np.float64)
        idx = int(np.argmin(np.abs(delays - float(delay))))
        if abs(delays[idx] - float(delay)) <= tol_val:
            e = entries[idx]
            return e, self.load_phase_map(e.path)

        raise FileNotFoundError(
            f"No delay within tol={tol_val} for requested delay={delay} in folder: {folder}"
        )

    def list_phase_maps(self, folder: Union[str, Path]) -> List[PhaseMapEntry]:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        entries: List[PhaseMapEntry] = []
        seen: Dict[float, Path] = {}

        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() != ".npy":
                continue

            m = self._DELAY_RE.match(p.name)
            if not m:
                continue  # ignore unrelated files

            delay = float(m.group("delay"))

            if delay in seen:
                raise ValueError(
                    f"Duplicate delay {delay} found in folder:\n"
                    f"  {folder}\n"
                    f"Files:\n"
                    f"  {seen[delay]}\n"
                    f"  {p}\n"
                    f"Expected exactly one phase map per delay."
                )

            seen[delay] = p
            entries.append(PhaseMapEntry(delay=delay, path=p))

        entries.sort(key=lambda e: e.delay)
        return entries

    def find_phase_map_pairs_by_delay(self) -> Tuple[List[Tuple[PhaseMapEntry, PhaseMapEntry]], Dict[str, List[PhaseMapEntry]]]:
        perp_list = self.list_phase_maps(self.perpendicular_dir)
        para_list = self.list_phase_maps(self.parallel_dir)

        # --- exact matching -------------------------------------------------
        perp_by_delay = {e.delay: e for e in perp_list}
        para_by_delay = {e.delay: e for e in para_list}

        exact_delays = sorted(perp_by_delay.keys() & para_by_delay.keys())

        pairs: List[Tuple[PhaseMapEntry, PhaseMapEntry]] = [
            (perp_by_delay[d], para_by_delay[d]) for d in exact_delays
        ]

        used_perp = {perp_by_delay[d] for d in exact_delays}
        used_para = {para_by_delay[d] for d in exact_delays}

        # leftovers after exact matching
        remaining_perp = [e for e in perp_list if e not in used_perp]
        remaining_para = [e for e in para_list if e not in used_para]

        # --- tolerant matching ----------------------------------------------
        if self.tol > 0.0 and remaining_perp and remaining_para:
            remaining_para_sorted = sorted(remaining_para, key=lambda e: e.delay)

            for p in list(remaining_perp):
                deltas = [abs(p.delay - q.delay) for q in remaining_para_sorted]
                idx = int(np.argmin(deltas))
                if deltas[idx] <= self.tol:
                    q = remaining_para_sorted.pop(idx)
                    pairs.append((p, q))
                    remaining_perp.remove(p)

                if not remaining_para_sorted:
                    break

        # --- unmatched ------------------------------------------------------
        unmatched = {
            "perpendicular": remaining_perp,
            "parallel": remaining_para,
        }

        return pairs, unmatched

    # Optional convenience
    def load_pair(self, pair: Tuple[PhaseMapEntry, PhaseMapEntry]) -> Tuple[PhaseMap, PhaseMap]:
        pe, qe = pair
        return self.load_phase_map(pe.path), self.load_phase_map(qe.path)

    def plot_phase_map_pair(self, pair, width_mm: Optional[float] = 6.0, depth_mm: Optional[float] = 5.0):
        perp_entry, para_entry = pair

        # Load with metadata
        phase_perp, meta_perp = self.load_phase_map_with_metadata(perp_entry.path)
        phase_para, meta_para = self.load_phase_map_with_metadata(para_entry.path)

        phase_perp = np.asarray(phase_perp, dtype=np.float32)
        phase_para = np.asarray(phase_para, dtype=np.float32)

        # Validate shapes (not required, but helpful)
        if phase_perp.ndim != 2 or phase_para.ndim != 2:
            raise ValueError("Phase maps must be 2D arrays")

        # Use metadata width if available; otherwise use parameter or default
        perp_width = meta_perp['roi_width_mm'] if meta_perp else width_mm
        para_width = meta_para['roi_width_mm'] if meta_para else depth_mm

        def _extent(arr: np.ndarray, x_length):
            h, w = arr.shape
            px_mm = float(x_length) / float(w)
            height_mm = h * px_mm
            # y=0 at nozzle edge (bottom) -> nozzle edge is last row -> put 0 at bottom
            return [0.0, float(x_length), 0.0, float(height_mm)]  # mm

        ext_perp = _extent(phase_perp, perp_width)
        ext_para = _extent(phase_para, para_width)

        # Display-only scaling: symmetric around 0 using robust percentile across BOTH maps
        combined = np.concatenate([phase_perp.ravel(), phase_para.ravel()])
        m = np.nanpercentile(np.abs(combined), float(98.0))
        m = max(float(m), 1e-12)
        vmin = -m
        vmax = +m

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        im0 = axes[0].imshow(
            phase_perp,
            origin="upper",  # row 0 at top
            extent=ext_perp,
            aspect="auto",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(f"Perpendicular slit\ndelay={perp_entry.delay:.6f}")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)  (0 = nozzle edge)")
        fig.colorbar(im0, ax=axes[0], label="Phase (rad)")

        im1 = axes[1].imshow(
            phase_para,
            origin="upper",
            extent=ext_para,
            aspect="auto",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(f"Parallel slit\ndelay={para_entry.delay:.6f}")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("y (mm)  (0 = nozzle edge)")
        fig.colorbar(im1, ax=axes[1], label="Phase (rad)")

        plt.show()

        return fig, axes

    def compute_depth_profile(
            self,
            delay: float,
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

        entry, phi = self.load_phase_map_by_delay(self.parallel_dir, delay, tol=tol)
        phi = np.asarray(phi, dtype=np.float32)

        # Try to load metadata for accurate scaling
        metadata = self.load_metadata(entry.path)
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
                        [0.1 * pmax_abs, x_peak - w/2, 0.5],  # lower bounds (always positive)
                        [2.0 * pmax_abs, x_peak + w/2, w]     # upper bounds
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

            plt.show()        # plot depth_mm vs y_mm
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

    def compute_density_from_phase_maps(
            self,
            *,
            min_depth_mm: float = 1e-6,
            max_height_mm: float = 6.0,
            perp_width_mm: float = 6,
            para_width_mm: float = 5,
            gas : str = "argon",
            debug_show: bool = False,
            selected_delay: Optional[float] = None,
            tol: Optional[float] = None,
            save_results: bool = True,
            save_dir: str | Path | None = None,
            y_profile_mm: float = 2.0,
    ) -> List[Dict[str, Any]]:

        """
        For each perpendicular/parallel phase-map pair:
          - compute depth profile from the parallel map via `compute_depth_profile`
          - compute density map for the perpendicular map using:
              n(y,x) = phi(y,x) / (K * L(y))
            where K is Gladstone–Dale (m^3 / number) and L is path length in meters.

        Returns list of dicts:
          {
            "delay": float,
            "perp_path": str,
            "para_path": str,
            "y_mm": np.ndarray,          # cropped y grid (mm)
            "x_mm": np.ndarray,          # x grid (mm)
            "depth_mm": np.ndarray,      # depth interpolated onto perp y grid (mm), cropped
            "density_map": np.ndarray,   # number density (m^-3) [or whatever your K implies]
            "meta": dict,
          }
        """

        # ---------------------------- Constants --------------------------------
        if gas == "helium":
            gladstone_dale = 1.96e-4
            M_kg_per_mol = 4.0026e-3
        elif gas == "argon":
            gladstone_dale = 1.57e-4
            M_kg_per_mol = 39.95e-3
        else:
            raise ValueError(f"Unsupported gas: {gas}. Supported gases: helium, argon.")
        wavelength_m = 532e-9
        N_A = 6.022e23  # 1/mol
        # ----------------------------------------------------------------------

        if save_results:
            if save_dir is None or str(save_dir).strip() == "":
                save_dir = ask_for_folder(
                    title="Select output folder for density results",
                    initialdir=r"C:/Users/vmlab/Documents/density_measurements",
                )
            parent = Path(save_dir)
            parent.mkdir(parents=True, exist_ok=True)

            # use perpendicular folder name
            perp_name = Path(self.perpendicular_dir).name
            run_dir = parent / perp_name

            run_dir.mkdir(parents=True, exist_ok=True)

        pairs, unmatched = self.find_phase_map_pairs_by_delay()
        if not pairs:
            return []

        def _prep_for_y0_at_nozzle(arr: np.ndarray) -> np.ndarray:
            # nozzle edge is bottom row in saved arrays
            return np.flipud(arr)

        results: List[Dict[str, Any]] = []

        # optional: restrict to one delay (tolerant)
        if selected_delay is not None:
            sel = float(selected_delay)
            tol_val = self.tol if tol is None else float(tol)

            def _close(a, b):
                return abs(float(a) - float(b)) <= tol_val

            filtered = []
            for (pe, qe) in pairs:
                if _close(pe.delay, sel) or _close(qe.delay, sel):
                    filtered.append((pe, qe))
            pairs_to_process = filtered
            if not pairs_to_process:
                # nothing matched; return empty (or raise if you prefer)
                return []
        else:
            pairs_to_process = pairs

        # ---------------------------- Main loop --------------------------------
        for (perp_e, para_e) in pairs_to_process:
            delay_val = float(perp_e.delay)  # matched delay

            # --- load & prep perpendicular phase map (y=0 at nozzle) ------------
            phi_perp_raw = np.asarray(self.load_phase_map(perp_e.path), dtype=np.float32)
            phi_perp = _prep_for_y0_at_nozzle(phi_perp_raw)
            if phi_perp.ndim != 2:
                raise ValueError(f"Perp phase map must be 2D: {perp_e.path}")

            h_perp, w_perp = phi_perp.shape
            
            # Try to load metadata for actual scaling
            meta_perp = self.load_metadata(perp_e.path)
            if meta_perp:
                actual_perp_width_mm = meta_perp['roi_width_mm']
            else:
                actual_perp_width_mm = float(perp_width_mm)
            
            px_mm_perp = float(actual_perp_width_mm) / float(w_perp)
            y_perp_mm_full = np.arange(h_perp, dtype=np.float32) * px_mm_perp
            x_perp_mm = np.linspace(0.0, float(actual_perp_width_mm), num=w_perp, endpoint=False, dtype=np.float32)

            # --- depth profile from parallel (already y=0 at nozzle in compute_depth_profile) ---
            y_para_mm, depth_para_mm, depth_entry, depth_debug = self.compute_depth_profile(
                delay_val,
                width_mm=float(para_width_mm),
                tol=tol,
            )
            y_para_mm = np.asarray(y_para_mm, dtype=np.float32)
            depth_para_mm = np.asarray(depth_para_mm, dtype=np.float32)

            # --- interpolate depth onto perp y grid -----------------------------
            # NOTE: np.interp does not handle NaNs nicely; mask them out.
            ok = np.isfinite(depth_para_mm) & np.isfinite(y_para_mm)
            if np.count_nonzero(ok) < 2:
                # not enough usable depth info; skip this delay
                continue

            depth_on_perp_mm_full = np.full_like(y_perp_mm_full, np.nan, dtype=np.float32)
            depth_on_perp_mm_full[:] = np.interp(
                y_perp_mm_full,
                y_para_mm[ok],
                depth_para_mm[ok],
                left=np.nan,
                right=np.nan,
            )

            # --- crop to y in [0.1, max_height_mm] ------------------------------
            y_min_mm = 0.2
            y_max_mm = float(max_height_mm)

            y_mask = (y_perp_mm_full >= y_min_mm) & (y_perp_mm_full <= y_max_mm)
            if not np.any(y_mask):
                continue

            y_perp_mm = y_perp_mm_full[y_mask]
            phi_perp_crop = phi_perp[y_mask, :]
            depth_mm = depth_on_perp_mm_full[y_mask]

            jump_thr = 1.6 * np.pi
            for i in range(phi_perp_crop.shape[0]):
                row = phi_perp_crop[i, :]
                d = np.diff(row)
                wrap_hits = np.where(np.abs(d) > float(jump_thr))[0]
                # if wrap_hits.size >= 1:
                #     phi_perp_crop[i, :] = np.unwrap(row).astype(np.float64)

            iy = int(np.argmin(np.abs(y_perp_mm - float(y_profile_mm))))
            profile = phi_perp_crop[iy, :]
            # plot profile for debug
            plt.figure(figsize=(7.5, 4.5), constrained_layout=True)
            plt.plot(x_perp_mm, profile, "-k", linewidth=1.8)
            plt.grid(True, alpha=0.3)
            plt.xlabel("x (mm)")
            plt.ylabel("number density (cm$^{-3}$)")
            plt.title(f"Density profile at y={y_perp_mm[iy]:.3f} mm\n"
                      f"delay={1000 * delay_val:.3f} ms")
            plt.show()

            # enforce min depth
            depth_mm = np.where(np.isfinite(depth_mm) & (depth_mm >= float(min_depth_mm)), depth_mm, np.nan)

            # --- density conversion --------------------------------------------
            #   rho = phi * wavelength / (2*pi*K*depth)
            #
            # IMPORTANT: Units:
            #   lambda in meters
            #   depth in meters
            #   K must be in m^3 / (number density unit)
            #
            # Here we compute n in whatever "number density" unit matches K.
            depth_m = depth_mm * 1e-3  # mm -> m
            rho_mass = np.full(phi_perp_crop.shape, np.nan, dtype=np.float64)  # kg/m^3
            n_cm3 = np.full(phi_perp_crop.shape, np.nan, dtype=np.float64)  # 1/cm^3
            denom = (2.0 * np.pi * float(gladstone_dale) * depth_m).astype(np.float64)

            density_map = np.full(phi_perp_crop.shape, np.nan, dtype=np.float64)
            good_rows = np.isfinite(denom) & (denom != 0.0)

            # broadcast denom over x
            if np.any(good_rows):
                rho_mass[good_rows, :] = -1 *(phi_perp_crop[good_rows, :].astype(np.float64) * wavelength_m) / denom[
                    good_rows, None]
                n_m3 = (rho_mass[good_rows, :] / M_kg_per_mol) * N_A
                n_cm3[good_rows, :] = n_m3 / 1e6

                n_cm3 = np.maximum(n_cm3, 0.0)

            meta = {
                "delay_matched": delay_val,
                "perp_width_mm": float(actual_perp_width_mm),
                "para_width_mm": float(para_width_mm),
                "px_mm_perp": float(px_mm_perp),
                "min_depth_mm": float(min_depth_mm),
                "y_crop_min_mm": float(y_min_mm),
                "y_crop_max_mm": float(y_max_mm),
                "depth_source_path": str(depth_entry.path),
                "depth_source_delay": float(depth_entry.delay),
            }

            results.append({
                "delay": delay_val,
                "perp_path": str(perp_e.path),
                "para_path": str(para_e.path),
                "y_mm": y_perp_mm,
                "x_mm": x_perp_mm,
                "depth_mm": depth_mm,
                "density_map": n_cm3,
                "meta": meta,
            })

            if save_results:
                assert run_dir is not None

                dname = f"delay_{delay_val:.6f}"
                out_d = run_dir / dname
                out_d.mkdir(parents=True, exist_ok=True)

                # --- save full density map --------------------------------------------
                np.save(out_d / "density_map.npy", n_cm3)
                np.save(out_d / "x_mm.npy", x_perp_mm)
                np.save(out_d / "y_mm.npy", y_perp_mm)

                # --- extract & save profile at y_profile_mm ----------------------------
                iy = int(np.argmin(np.abs(y_perp_mm - float(y_profile_mm))))
                profile = n_cm3[iy, :]

                np.save(out_d / f"profile_y{y_perp_mm[iy]:.3f}mm.npy", profile)

                # --- metadata ----------------------------------------------------------
                meta_out = {**meta, "y_profile_mm": float(y_perp_mm[iy]), "iy_profile": int(iy), "delay": delay_val,}
                np.save(out_d / "meta.npy", meta_out)

            # optional quick debug plot of density map
            if debug_show:
                y_target_mm = y_profile_mm
                i2 = int(np.argmin(np.abs(y_perp_mm - y_target_mm)))

                prof = n_cm3[i2, :]
                plt.figure(figsize=(7.5, 4.5), constrained_layout=True)
                plt.plot(x_perp_mm, prof, "-k", linewidth=1.8)
                plt.grid(True, alpha=0.3)
                plt.xlabel("x (mm)")
                plt.ylabel("Number density (cm$^{-3}$)")
                plt.title(f"Density profile at y={y_perp_mm[i2]:.3f} mm\n"
                          f"delay={1000 * delay_val:.3f} ms")
                plt.show()

                # build extent for cropped map
                extent = [0.0, float(actual_perp_width_mm), float(y_perp_mm[0]), float(y_perp_mm[-1])]

                plt.figure(figsize=(6.5, 5), constrained_layout=True)
                # robust scaling for display
                m = np.nanpercentile(np.abs(n_cm3), 98.0)
                m = max(float(m), 1e-30)
                plt.imshow(n_cm3, origin="lower", extent=extent, aspect="auto")
                plt.axhline(y=y_target_mm, linestyle="--", linewidth=1.5)
                plt.colorbar(label="density 1/cm³")
                plt.title(f"Density map — delay={1000 * delay_val:.6f}ms")
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm) (0 = nozzle edge)")
                plt.show()

        return results


    def shock_length_threshold_method(
            self,
            density_results: list,
            *,
            delay: float,
            y_target_mm: float = 2.0,
            baseline_window_mm: float = 2.0,
            smooth_win_px: int = 11,  # light smoothing helps locate minima/crossing
            min_prominence_frac: float = 0.02,  # reject tiny noisy minima (fraction of peak)
            debug_plot: bool = True,
    ):
        """
        Noise-robust shock-length estimator:

        - take profile at y_target_mm
        - find peak (x_max, n_max)
        - find first local minimum to the left of peak (x_min)
        - baseline window = [x_min - baseline_window_mm, x_min]
        - threshold n_thr = median(profile in baseline window)
        - x_pre = first x (moving left->peak) where profile crosses upward above n_thr
          *closest to the peak* crossing is used (more stable)
        - slope = (n_max - n_thr)/(x_max - x_pre)
        - shock_length = n_max / |slope|

        Returns: shock_length_mm, debug dict
        """

        def _find_peak_and_left_minimum(prof_s, *,
                                        peak_prom_frac=0.05, min_prom_frac=0.02, min_distance_px=5):

            nmax_est = np.nanmax(prof_s)
            if not np.isfinite(nmax_est) or nmax_est <= 0:
                raise ValueError("Profile peak not finite/positive")

            # --- main peak (use prominence to avoid picking noise spikes)
            peaks, _ = find_peaks(
                prof_s,
                prominence=peak_prom_frac * nmax_est,
                distance=min_distance_px,
            )
            imax = int(np.nanargmax(prof_s)) if peaks.size == 0 else int(peaks[np.argmax(prof_s[peaks])])

            # --- minima: peaks of inverted signal
            inv = -prof_s
            mins, _ = find_peaks(
                inv,
                prominence=min_prom_frac * nmax_est,
                distance=min_distance_px,
            )
            # keep only minima left of peak
            mins = mins[mins < imax]
            if mins.size == 0:
                raise ValueError("No robust local minimum found left of peak; try more smoothing or lower prominence")

            # choose the closest minimum to the peak
            imin = int(mins[np.argmax(mins)])
            return imax, imin
        # --- find result by delay --------------------------------------------
        tol = float(getattr(self, "tol", 1e-6))
        r = None
        for rr in density_results:
            if abs(float(rr["delay"]) - float(delay)) <= tol:
                r = rr
                break
        if r is None:
            raise KeyError(f"No density result found for delay={delay} (tol={tol})")

        x = np.asarray(r["x_mm"], dtype=np.float64)
        y = np.asarray(r["y_mm"], dtype=np.float64)
        nmap = np.asarray(r["density_map"], dtype=np.float64)

        iy = int(np.argmin(np.abs(y - float(y_target_mm))))
        y_used = float(y[iy])

        prof = nmap[iy, :].copy()
        finite = np.isfinite(prof) & np.isfinite(x)
        if np.count_nonzero(finite) < 10:
            raise ValueError("Not enough finite points in profile")

        # --- fill NaNs for smoothing / crossing logic -------------------------
        prof_f = prof.copy()
        nanmask = ~np.isfinite(prof_f)
        if np.any(nanmask):
            prof_f[nanmask] = np.interp(x[nanmask], x[~nanmask], prof_f[~nanmask])

        # --- light smoothing (only for feature detection) ---------------------
        prof_s = prof_f.copy()
        if smooth_win_px and smooth_win_px > 1:
            k = int(smooth_win_px)
            kernel = np.ones(k, dtype=np.float64) / k
            prof_s = np.convolve(prof_s, kernel, mode="same")

        imax, imin = _find_peak_and_left_minimum(prof_s)

        x_max, x_min = float(x[imax]), float(x[imin])

        # --- threshold: median in [x_min - window, x_min] ---------------------
        x0 = x_min - float(baseline_window_mm)
        win_mask = (x >= x0) & (x <= x_min)
        if np.count_nonzero(win_mask) < 5:
            raise ValueError("Baseline window too small/empty; increase window or check x range")

        n_thr = float(np.median(prof_f[win_mask]))  # use unsmoothed-filled for median robustness

        # --- x_pre: last upward crossing before peak --------------------------
        # We want the crossing *closest to the peak* (more stable):
        # Find indices where prof crosses from below->above threshold, then pick the last one < imax.
        above = prof_s >= n_thr
        cross_idxs = np.where((~above[:-1]) & (above[1:]))[0] + 1  # index of first above sample
        cross_idxs = cross_idxs[cross_idxs < imax]
        if cross_idxs.size == 0:
            raise ValueError("No threshold crossing found before peak; threshold may be too high")

        ic = int(cross_idxs[-1])  # closest crossing to peak

        # interpolate crossing x between ic-1 and ic
        x1, x2 = x[ic - 1], x[ic]
        y1, y2 = prof_s[ic - 1], prof_s[ic]
        if y2 == y1:
            x_pre = float(x2)
        else:
            frac = (n_thr - y1) / (y2 - y1)
            x_pre = float(x1 + frac * (x2 - x1))

        dx = x_max - x_pre
        if dx <= 0:
            raise ValueError("Computed x_pre is not left of x_max; check logic/threshold")

        slope = (n_max - n_thr) / dx  # (cm^-3)/mm
        shock_length_mm = n_max / abs(slope)

        debug = dict(
            delay=float(delay),
            y_used_mm=y_used,
            x_max_mm=x_max,
            n_max_cm3=n_max,
            x_min_mm=x_min,
            n_min_cm3=n_min,
            baseline_window_mm=float(baseline_window_mm),
            n_thr_cm3=n_thr,
            x_pre_mm=x_pre,
            slope_cm3_per_mm=float(slope),
            shock_length_mm=float(shock_length_mm),
        )

        if debug_plot:
            fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
            ax.plot(x, prof, alpha=0.25, label="raw")
            ax.plot(x, prof_s, "-k", linewidth=2, label="smoothed (for detection)")

            # peak
            ax.plot([x_max], [n_max], "o", markersize=8, label="n_max")

            # chosen local min
            ax.plot([x_min], [n_min], "o", markersize=7, label="local min (left of peak)")

            # baseline window + threshold
            ax.axvspan(x0, x_min, alpha=0.15, label="baseline window")
            ax.hlines(n_thr, x0, x_min, linestyles="--", linewidth=2, label="threshold (median)")

            # x_pre crossing
            ax.axvline(x_pre, linestyle=":", linewidth=2, label="x_pre (threshold crossing)")

            # secant used for slope (from threshold at x_pre to peak)
            ax.plot([x_pre, x_max], [n_thr, n_max], "--", linewidth=2, label="secant for slope")

            txt = (
                f"y={y_used:.3f} mm (target {y_target_mm:.3f})\n"
                f"x_pre={x_pre:.3f} mm, n_thr={n_thr:.3e} cm⁻³\n"
                f"x_max={x_max:.3f} mm, n_max={n_max:.3e} cm⁻³\n"
                f"|dn/dx|≈{abs(slope):.3e} cm⁻³/mm\n"
                f"shock_length={shock_length_mm:.3f} mm"
            )
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left")

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("Number density (cm$^{-3}$)")
            ax.set_title(f"Shock length (threshold method) — delay={1000 * float(delay):.3f} ms")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            plt.show()

        return shock_length_mm, debug

    def plot_depth_profile(self,delay: float):
        """
        Debugging plot of depth profile computed from parallel slit phase map at given delay.
        """
        y_mm, depth_mm, entry, debug = self.compute_depth_profile(delay)

        plt.figure(figsize=(5.5, 5))
        plt.plot(depth_mm, y_mm, "-k", linewidth=1.5)
        plt.xlabel("Effective thickness (mm)  [FWHM across x]")
        plt.ylabel("y (mm)  (0 = nozzle edge)")
        plt.title(f"Depth vs height (parallel) — delay={entry.delay:.6f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return y_mm, depth_mm, debug

    def plot_phase_map_perp_by_delay(
            self,
            delay: float,
            *,
            width_mm: Optional[float] = 6.0,
            tol: Optional[float] = None,
            nozzle_row: str = "last",  # "last" if nozzle edge is last row in saved array
            robust_pct: float = 98.0,  # display scaling percentile
            cmap: str = "seismic",
            vmin: Optional[float] = None,  # override if you want fixed scaling
            vmax: Optional[float] = None,  # override if you want fixed scaling
            title: Optional[str] = None,
            ax=None,
            show: bool = True,
    ):
        """
        Debug: plot PERPENDICULAR phase map for a given delay from self.perpendicular_dir.

        - Uses load_phase_map_by_delay(self.perpendicular_dir, delay, tol=...)
        - Enforces y=0 at nozzle edge by flipping if nozzle_row=="last"
        - Uses metadata scaling if available, otherwise falls back to width_mm parameter.
        """
        entry, phi = self.load_phase_map_by_delay(self.perpendicular_dir, delay, tol=tol)
        phi = np.asarray(phi, dtype=np.float32)

        if phi.ndim != 2:
            raise ValueError(f"Phase map must be 2D, got shape={phi.shape} for {entry.path}")

        # --- enforce y convention: row 0 = nozzle edge -------------------------
        if nozzle_row == "last":
            phi_proc = np.flipud(phi)
        elif nozzle_row == "first":
            phi_proc = phi
        else:
            raise ValueError("nozzle_row must be 'first' or 'last'")

        h, w = phi_proc.shape
        
        # Try to load metadata for actual scaling
        metadata = self.load_metadata(entry.path)
        if metadata:
            actual_width_mm = metadata['roi_width_mm']
        else:
            actual_width_mm = width_mm  # Fallback to parameter or default
        
        px_mm = float(actual_width_mm) / float(w)
        height_mm = float(h) * px_mm

        extent = [0.0, float(actual_width_mm), 0.0, float(height_mm)]  # x in [0,width], y in [0,height]

        # --- robust display scaling (symmetric around 0) ------------------------
        if vmin is None or vmax is None:
            m = np.nanpercentile(np.abs(phi_proc.ravel()), float(robust_pct))
            m = max(float(m), 1e-12)
            vmin = -m
            vmax = +m

        # --- plot --------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2), constrained_layout=True)
        else:
            fig = ax.figure

        im = ax.imshow(
            phi_proc,
            origin="lower",  # since we flipped (if needed), y=0 is now at bottom -> origin lower
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ttl = title or f"Perpendicular phase map — delay={entry.delay:.6f}\n{entry.path.name}"
        ax.set_title(ttl)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)  (0 = nozzle edge)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Phase (rad)")

        if show:
            plt.show()

        return fig, ax, entry, phi_proc

class CylindricalNozzleAnalysis:
    """
    Analysis helper for cylindrical nozzle interferometry where phase maps are from a
    single axially symmetric phasemap (e.g., computed with cylindrical symmetry assumption).
    
    Unlike SlitNozzleAnalysis which needs two orthogonal phase maps, CylindricalNozzleAnalysis
    uses one phase map and inverts it via Abel transform to get radial density profiles.
    
    Expected filename format:
        delay_<value>.npy
        delay_<value>_<occurrence>.npy
    """
    
    _DELAY_RE = re.compile(
        r"^delay_(?P<delay>[+-]?\d+(?:\.\d+)?)\.npy$",
        re.IGNORECASE,
    )

    def __init__(self, cylindrical_dir: str | Path | None = None, tol: float = 1e-6):
        """
        Initialize CylindricalNozzleAnalysis.
        
        Args:
            cylindrical_dir: Path to folder containing cylindrical phase maps (.npy files).
                           If None, prompts user to select folder.
            tol: Tolerance for matching delays (float).
        """
        if cylindrical_dir is None or str(cylindrical_dir).strip() == "":
            cylindrical_dir = ask_for_folder(
                title="Select CYLINDRICAL phase-map folder",
                initialdir=r"C:/Users/vmlab/Documents/data/phase_maps"
            )

        self.cylindrical_dir = Path(cylindrical_dir)
        self.tol = float(tol)

    @staticmethod
    def load_phase_map(path: Union[str, Path]) -> PhaseMap:
        """Load a .npy phase map file."""
        path = Path(path)
        arr = np.load(str(path))
        return np.asarray(arr)

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> Optional[Dict[str, float]]:
        """Load metadata JSON for a phase map. Returns None if file doesn't exist."""
        path = Path(path)
        metadata_path = path.with_suffix('.json')
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def load_phase_map_with_metadata(self, path: Union[str, Path]) -> Tuple[PhaseMap, Optional[Dict[str, float]]]:
        """Load phase map and try to load accompanying metadata JSON."""
        phase_map = self.load_phase_map(path)
        metadata = self.load_metadata(path)
        return phase_map, metadata

    def list_phase_maps(self, folder: Union[str, Path]) -> List[PhaseMapEntry]:
        """List all phase maps in a folder, sorted by delay."""
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        entries: List[PhaseMapEntry] = []
        seen: Dict[float, Path] = {}

        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() != ".npy":
                continue

            m = self._DELAY_RE.match(p.name)
            if not m:
                continue

            delay = float(m.group("delay"))

            if delay in seen:
                raise ValueError(
                    f"Duplicate delay {delay} found in folder:\n"
                    f"  {folder}\n"
                    f"Files:\n"
                    f"  {seen[delay]}\n"
                    f"  {p}\n"
                    f"Expected exactly one phase map per delay."
                )

            seen[delay] = p
            entries.append(PhaseMapEntry(delay=delay, path=p))

        entries.sort(key=lambda e: e.delay)
        return entries

    def load_phase_map_by_delay(self, folder: Union[str, Path], delay: float,
        *, tol: Optional[float] = None) -> Tuple[PhaseMapEntry, PhaseMap]:
        """
        Load phase map by matching delay value (exact or tolerant).
        
        Args:
            folder: Path to folder containing phase maps
            delay: Target delay to match
            tol: Tolerance for delay matching (uses self.tol if None)
            
        Returns:
            (PhaseMapEntry, phase_map_array)
        """
        folder = Path(folder)
        entries = self.list_phase_maps(folder)

        # exact match first
        for e in entries:
            if e.delay == float(delay):
                return e, self.load_phase_map(e.path)

        # tolerant match
        tol_val = self.tol if tol is None else float(tol)
        if tol_val <= 0.0:
            raise FileNotFoundError(f"No exact delay={delay} in folder: {folder}")

        delays = np.array([e.delay for e in entries], dtype=np.float64)
        idx = int(np.argmin(np.abs(delays - float(delay))))
        if abs(delays[idx] - float(delay)) <= tol_val:
            e = entries[idx]
            return e, self.load_phase_map(e.path)

        raise FileNotFoundError(
            f"No delay within tol={tol_val} for requested delay={delay} in folder: {folder}"
        )

    @staticmethod
    def abel_invert_simple(phase_map: np.ndarray, px_mm: float = 1.0, axis: int = 1) -> np.ndarray:
        """
        Abel inversion using physical coordinates (mm) with symmetry averaging.
        
        Assumes cylindrical symmetry about the CENTER of the phase map.
        Strategy:
        1. Split into left and right halves around center
        2. Average the two halves (leveraging symmetry to reduce noise)
        3. Apply Abel inversion to the averaged half
        4. Mirror to create full symmetric inverted image
        
        Args:
            phase_map: 2D array where axis 1 (x) is the radial coordinate
            px_mm: Pixel size in mm (needed for proper physical integration)
            axis: axis along which to perform inversion (default 1 = x-axis)
            
        Returns:
            Inverted 2D array of same shape
        """
        if phase_map.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {phase_map.shape}")
        
        h, w = phase_map.shape
        inverted_total = np.zeros_like(phase_map, dtype=np.float64)
        
        # Find center
        center = w // 2
        
        # For each y-row: average halves, invert, then mirror
        for i in range(h):
            line = phase_map[i, :].astype(np.float64)
            
            # Split into halves
            left_radial = line[:center + 1][::-1]
            right_radial = line[center:]

            # Use only the common radial range.
            n_radial = min(len(left_radial), len(right_radial))
            left_radial = left_radial[:n_radial]
            right_radial = right_radial[:n_radial]

            # Even/symmetric part: this is what Abel inversion should see.
            radial_line = 0.5 * (left_radial + right_radial)

            if n_radial > 1:
                inverted_radial = CylindricalNozzleAnalysis._abel_line(
                    radial_line,
                    px_mm=px_mm
                )
            else:
                inverted_radial = radial_line.copy()

            # Put the radial result back into full x-space.
            # Left side needs to be reversed back: outer ... center
            inverted_total[i, center - n_radial + 1:center + 1] = inverted_radial[::-1]

            # Right side: center ... outer
            inverted_total[i, center:center + n_radial] = inverted_radial

        return inverted_total.astype(np.float32)

        #     # Flip right half to align with left half for averaging
        #     right_half_flipped = np.flip(right_half)
        #
        #     # Make equal length for averaging
        #     min_len = min(len(left_half), len(right_half_flipped))
        #     left_trimmed = left_half[-min_len:] if len(left_half) > min_len else left_half
        #     right_trimmed = right_half_flipped[-min_len:] if len(right_half_flipped) > min_len else right_half_flipped
        #
        #     # Average the two halves
        #     averaged_half = (left_trimmed + right_trimmed) / 2.0
        #
        #     # Apply Abel inversion to averaged half
        #     if len(averaged_half) > 1:
        #         inverted_half = CylindricalNozzleAnalysis._abel_line(averaged_half, px_mm=px_mm)
        #     else:
        #         inverted_half = averaged_half.copy()
        #
        #     # Create full image by mirroring: flip to get right side, concatenate
        #     inverted_right = np.flip(inverted_half)
        #
        #     # Reconstruct full row with symmetry
        #     if w % 2 == 0:
        #         # Even width: mirror exactly
        #         inverted_total[i, :center] = inverted_half
        #         inverted_total[i, center:] = inverted_right
        #     else:
        #         # Odd width: center pixel, then mirror
        #         inverted_total[i, :center] = inverted_half
        #         inverted_total[i, center:] = inverted_right
        #
        # return inverted_total.astype(np.float32)

    @staticmethod
    def _abel_line(line: np.ndarray, px_mm: float = 1.0) -> np.ndarray:
        """
        Abel inversion of a 1D line using numerical integration in physical coordinates.
        
        Assumes line[r] is the line-of-sight integral (phase map data) at radius r.
        Returns inverted[r] as the local density at radius r.
        
        Uses the standard Abel inversion formula:
            f(r) = -1/π * ∫_r^R (dF/ds) / √(s²-r²) ds
        
        where:
            F(s) = line[s] (measured line-of-sight integral)
            dF/ds = numerical derivative of F with respect to s (in physical coordinates)
            f(r) = inverted local density
            
        Args:
            line: 1D array of phase map values
            px_mm: Pixel size in mm (for proper physical integration)
        """
        n = len(line)
        inverted = np.zeros(n, dtype=np.float64)
        
        # Compute numerical derivative dF/ds using central differences in physical space
        dF_ds = np.zeros(n, dtype=np.float64)
        dF_ds[0] = (line[1] - line[0]) / px_mm  # Forward difference at start
        dF_ds[-1] = (line[-1] - line[-2]) / px_mm  # Backward difference at end
        dF_ds[1:-1] = (line[2:] - line[:-2]) / (2.0 * px_mm)  # Central difference in middle
        
        # Apply Abel inversion formula at each radius r
        for r in range(n - 1):
            # Convert pixel index r to physical coordinate in mm
            r_mm = float(r) * px_mm
            
            # Create arrays for integration from r to n-1
            s_indices = np.arange(r + 1, n, dtype=np.int32)
            s_mm = s_indices.astype(np.float64) * px_mm  # Convert to physical coordinates
            
            # Compute the weighting factors: 1 / √(s² - r²)
            # All values in mm for consistent units
            s_squared = s_mm ** 2
            r_squared = r_mm ** 2
            denominators = np.sqrt(np.maximum(s_squared - r_squared, 1e-10))
            weights = 1.0 / denominators
            
            # Integrate: ∫_r^R (dF/ds) / √(s²-r²) ds
            dF_vals = dF_ds[s_indices]
            integrand = dF_vals * weights
            
            # Use trapezoidal rule for numerical integration (with physical spacing)
            integral = np.trapz(integrand, x=s_mm)
            
            # Apply Abel inversion formula coefficient
            inverted[r] = -integral / np.pi
        
        return inverted

    def compute_density_from_cylindrical_phase_map(
            self,
            delay: float,
            *,
            width_mm: float = 5.0,
            gas: str = "argon",
            tol: Optional[float] = None,
            save_results: bool = False,
            save_dir: str | Path | None = None,
            y_profiles_mm: Optional[list[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute density from a single cylindrical phase map using Abel inversion.
        
        Physical model:
          - Cylindrical symmetry: density ρ(r, y, z) depends only on (r, y)
          - Line-of-sight integral: φ(x, y) = κ * ∫ρ(r, y) dr (from r=0 to x)
          - Abel inversion recovers local ρ(x, y) from φ(x, y)
        
        Args:
            delay: Delay value to match phase maps
            width_mm: Physical width of the measured region (mm)
            gas: Gas type ("argon" or "helium")
            debug_show: Whether to show debug plots
            tol: Tolerance for delay matching
            save_results: Whether to save results to disk
            save_dir: Output directory for results
            y_profile_mm: Y position for profile extraction
            
        Returns:
            Dictionary with keys:
                "delay": float,
                "path": str,
                "y_mm": np.ndarray,
                "x_mm": np.ndarray,
                "phase_map": np.ndarray,
                "density_map": np.ndarray,
                "meta": dict,
        """
        
        # Constants
        if gas == "helium":
            gladstone_dale = 1.96e-4
            M_kg_per_mol = 4.0026e-3
        elif gas == "argon":
            gladstone_dale = 1.57e-4
            M_kg_per_mol = 39.95e-3
        else:
            raise ValueError(f"Unsupported gas: {gas}. Supported: helium, argon.")
        wavelength_m = 532e-9
        N_A = 6.022e23

        if y_profiles_mm is None:
            y_profiles_mm = [1, 2, 3, 4, 5, 6]

        y_profiles_mm = [float(y) for y in y_profiles_mm]

        # Load phase map
        entry, phase_map_raw = self.load_phase_map_by_delay(self.cylindrical_dir, delay, tol=tol)
        phase_map = np.asarray(phase_map_raw, dtype=np.float32)
        
        if phase_map.ndim != 2:
            raise ValueError(f"Phase map must be 2D, got shape {phase_map.shape}")
        
        h, w = phase_map.shape
        
        # Load metadata for scaling
        metadata = self.load_metadata(entry.path)
        if metadata:
            actual_width_mm = metadata['roi_width_mm']
        else:
            actual_width_mm = float(width_mm)
        
        px_mm = float(actual_width_mm) / float(w)
        y_mm_full = np.arange(h, dtype=np.float32) * px_mm
        x_mm = np.linspace(0.0, float(actual_width_mm), num=w, endpoint=False, dtype=np.float32)
        
        # Enforce y=0 at nozzle edge (assuming last row = nozzle edge)
        phase_map_proc = np.flipud(phase_map)

        jump_thr = 1.6 * np.pi
        for i in range(phase_map_proc.shape[0]):
            row = phase_map_proc[i, :]
            d = np.diff(row)
            wrap_hits = np.where(np.abs(d) > float(jump_thr))[0]
            if wrap_hits.size >= 1:
                phase_map_proc[i, :] = np.unwrap(row).astype(np.float64)

        # Apply Abel inversion to get local density (in phase units)
        # Pass physical pixel size for proper dimensional analysis
        phase_inverted = self.abel_invert_simple(phase_map_proc, px_mm=px_mm, axis=1)
        phase_inverted_per_m = phase_inverted.astype(np.float64) * 1e3  # rad/mm -> rad/m

        # Convert phase to density
        # ρ = φ * λ / (2π * K)
        # Compute density map
        density_map = np.full(phase_map_proc.shape, np.nan, dtype=np.float64)
        denom = 2.0 * np.pi * float(gladstone_dale)
        
        # Simple conversion: phase -> number density
        density_map = phase_inverted_per_m * wavelength_m / denom
        n_cm3 = (density_map / (M_kg_per_mol / N_A)) / 1e6  # convert to cm^-3
        
        # Handle non-physical densities
        n_finite = n_cm3[np.isfinite(n_cm3)]
        if len(n_finite) > 0:
            n_min_original = float(np.min(n_finite))
            n_negative_count = np.sum(n_cm3 < 0.0)
            n_total_valid = np.sum(np.isfinite(n_cm3))
            
            if n_negative_count > 0:
                if PRINT_REPLIES:
                    print(f"[Density clipping] Found {n_negative_count}/{n_total_valid} negative density points")
                    print(f"  Original min: {n_min_original:.3e} cm⁻³")
        
        # Clip negative densities to 0
        n_cm3 = np.maximum(n_cm3, 0.0)

        result = {
            "delay": float(entry.delay),
            "path": str(entry.path),
            "y_mm": y_mm_full,
            "x_mm": x_mm,
            "phase_map": phase_map_proc,
            "density_map": n_cm3,
            "y_profiles_mm": y_profiles_mm,
            "meta": {
                "width_mm": float(actual_width_mm),
                "px_mm": float(px_mm),
                "gas": gas,
                "gladstone_dale": float(gladstone_dale),
            }
        }

        if save_results:
            if save_dir is None or str(save_dir).strip() == "":
                save_dir = ask_for_folder(
                    title="Select output folder for density results",
                    initialdir=r"C:/Users/vmlab/Documents/density_measurements"
                )
            
            save_dir = Path(save_dir).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            cyl_name = Path(self.cylindrical_dir).name
            dname = f"delay_{entry.delay:.6f}"
            out_d = save_dir / cyl_name / dname
            out_d.mkdir(parents=True, exist_ok=True)
            
            np.save(out_d / "density_map.npy", n_cm3)
            np.save(out_d / "phase_map.npy", phase_map_proc)
            np.save(out_d / "x_mm.npy", x_mm)
            np.save(out_d / "y_mm.npy", y_mm_full)

            # Extract and save profiles
            saved_profiles = {}

            for y_profile_mm in y_profiles_mm:
                iy = int(np.argmin(np.abs(y_mm_full - float(y_profile_mm))))
                profile = n_cm3[iy, :]

                y_actual = float(y_mm_full[iy])
                saved_profiles[y_profile_mm] = {
                    "requested_y_mm": float(y_profile_mm),
                    "actual_y_mm": y_actual,
                    "row_index": int(iy),
                    "filename": f"profile_y{y_actual:.3f}mm.npy",
                }

                np.save(out_d / f"profile_y{y_actual:.3f}mm.npy", profile)

            # Save metadata
            meta_out = {
                **result["meta"],
                "y_profiles_mm": y_profiles_mm,
                "saved_profiles": saved_profiles,
                "delay": float(entry.delay),
            }
            np.save(out_d / "meta.npy", meta_out)


        return result

    def debug_plot_phase_map(
            self,
            delay: float,
            *,
            width_mm: Optional[float] = None,
            tol: Optional[float] = None,
            nozzle_row: str = "last",
            robust_pct: float = 98.0,
            cmap: str = "seismic",
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            title: Optional[str] = None,
            show: bool = True,
    ):
        """
        Debug function: Plot cylindrical phase map for a given delay.
        
        Args:
            delay: Delay value to plot
            width_mm: Physical width (mm) - uses metadata if available
            tol: Tolerance for delay matching
            nozzle_row: "last" or "first" to indicate nozzle edge position
            robust_pct: Percentile for display scaling
            cmap: Colormap name
            vmin, vmax: Override display range
            title: Custom title
            show: Whether to call plt.show()
        """
        entry, phase_map = self.load_phase_map_by_delay(self.cylindrical_dir, delay, tol=tol)
        phase_map = np.asarray(phase_map, dtype=np.float32)
        
        if phase_map.ndim != 2:
            raise ValueError(f"Phase map must be 2D, got shape {phase_map.shape}")
        
        # Enforce y convention
        if nozzle_row == "last":
            phase_proc = np.flipud(phase_map)
        elif nozzle_row == "first":
            phase_proc = phase_map
        else:
            raise ValueError("nozzle_row must be 'first' or 'last'")
        
        h, w = phase_proc.shape
        
        # Load metadata for scaling
        metadata = self.load_metadata(entry.path)
        if metadata:
            actual_width_mm = metadata['roi_width_mm']
        else:
            actual_width_mm = width_mm if width_mm else 5.0
        
        px_mm = float(actual_width_mm) / float(w)
        height_mm = float(h) * px_mm
        
        extent = [0.0, float(actual_width_mm), 0.0, float(height_mm)]

        jump_thr = 1.6 * np.pi
        for i in range(phase_proc.shape[0]):
            row = phase_proc[i, :]
            d = np.diff(row)
            wrap_hits = np.where(np.abs(d) > float(jump_thr))[0]
            if wrap_hits.size >= 1:
                phase_proc[i, :] = np.unwrap(row).astype(np.float64)

        # Display scaling
        if vmin is None or vmax is None:
            m = np.nanpercentile(np.abs(phase_proc.ravel()), float(robust_pct))
            m = max(float(m), 1e-12)
            vmin = -m
            vmax = +m
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2), constrained_layout=True)
        
        im = ax.imshow(
            phase_proc,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        
        ttl = title or f"Cylindrical phase map — delay={entry.delay:.6f}\n{entry.path.name}"
        ax.set_title(ttl)
        ax.set_xlabel("x (mm)  (radial)")
        ax.set_ylabel("y (mm)  (0 = nozzle edge)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Phase (rad)")
        
        if show:
            plt.show()
        
        return fig, ax, entry, phase_proc

    def debug_abel_inversion(
            self,
            delay: float,
            *,
            width_mm: Optional[float] = None,
            tol: Optional[float] = None,
            nozzle_row: str = "last",
            robust_pct: float = 98.0,
            show: bool = True,
    ):
        """
        Debug function: Visualize the Abel inversion process.
        
        Plots the original phase map and the inverted result side-by-side,
        allowing inspection of how the Abel transform changes the data.
        
        Args:
            delay: Delay value to analyze
            width_mm: Physical width (mm) - uses metadata if available
            tol: Tolerance for delay matching
            nozzle_row: "last" or "first" to indicate nozzle edge position
            robust_pct: Percentile for display scaling
            show: Whether to call plt.show()
            
        Returns:
            fig, axes, entry, phase_map, inverted_map
        """
        # Load phase map
        entry, phase_map = self.load_phase_map_by_delay(self.cylindrical_dir, delay, tol=tol)
        phase_map = np.asarray(phase_map, dtype=np.float32)
        
        if phase_map.ndim != 2:
            raise ValueError(f"Phase map must be 2D, got shape {phase_map.shape}")
        
        # Enforce y convention
        if nozzle_row == "last":
            phase_proc = np.flipud(phase_map)
        elif nozzle_row == "first":
            phase_proc = phase_map
        else:
            raise ValueError("nozzle_row must be 'first' or 'last'")
        
        h, w = phase_proc.shape
        
        # Load metadata for scaling
        metadata = self.load_metadata(entry.path)
        if metadata:
            actual_width_mm = metadata['roi_width_mm']
        else:
            actual_width_mm = width_mm if width_mm else 5.0
        
        px_mm = float(actual_width_mm) / float(w)
        height_mm = float(h) * px_mm
        extent = [0.0, float(actual_width_mm), 0.0, float(height_mm)]
        
        # Apply Abel inversion
        inverted_map = self.abel_invert_simple(phase_proc, px_mm=px_mm, axis=1)
        
        # Display scaling for phase
        m_phase = np.nanpercentile(np.abs(phase_proc.ravel()), float(robust_pct))
        m_phase = max(float(m_phase), 1e-12)
        
        # Display scaling for inverted (may have different range)
        m_inverted = np.nanpercentile(np.abs(inverted_map.ravel()), float(robust_pct))
        m_inverted = max(float(m_inverted), 1e-12)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        
        # Original phase map
        im0 = axes[0].imshow(
            phase_proc,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="seismic",
            vmin=-m_phase,
            vmax=+m_phase,
        )
        axes[0].set_title(f"Original Phase Map\ndelay={entry.delay:.6f}")
        axes[0].set_xlabel("x (mm)  (radial)")
        axes[0].set_ylabel("y (mm)  (0 = nozzle edge)")
        cb0 = fig.colorbar(im0, ax=axes[0], label="Phase (rad)")
        
        # Abel inverted result
        im1 = axes[1].imshow(
            inverted_map,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=-m_inverted,
            vmax=+m_inverted,
        )
        axes[1].set_title(f"After Abel Inversion\npx_mm={px_mm:.4f}")
        axes[1].set_xlabel("x (mm)  (radial)")
        axes[1].set_ylabel("y (mm)  (0 = nozzle edge)")
        cb1 = fig.colorbar(im1, ax=axes[1], label="Inverted value")
        
        if show:
            plt.show()
        
        return fig, axes, entry, phase_proc, inverted_map

    def debug_plot_density_map(
            self,
            delay: float,
            *,
            width_mm: float = 5.0,
            gas: str = "argon",
            tol: Optional[float] = None,
            y_profile_mm: float = 2.0,
            robust_pct: float = 98.0,
            show: bool = True,
    ):
        """
        Debug function: Compute density and plot both profile and 2D map.
        
        This function handles the full computation pipeline:
        - Loads phase map for given delay
        - Applies Abel inversion
        - Converts to number density
        - Plots 1D profile and 2D density map
        
        Args:
            delay: Delay value to analyze
            width_mm: Physical width of measurement region (mm)
            gas: Gas type ("argon" or "helium")
            tol: Tolerance for delay matching
            y_profile_mm: Y position for 1D profile extraction
            robust_pct: Percentile for display scaling
            show: Whether to call plt.show()
            
        Returns:
            fig1, fig2, result (where result is the density computation result dict)
        """
        # Compute density from phase map
        result = self.compute_density_from_cylindrical_phase_map(
            delay=delay,
            width_mm=width_mm,
            gas=gas,
            tol=tol,
        )
        
        # Extract result data
        delay_val = result["delay"]
        y_mm = result["y_mm"]
        x_mm = result["x_mm"]
        density_map = result["density_map"]
        
        # Print density statistics
        if PRINT_REPLIES:
            valid_density = density_map[np.isfinite(density_map)]
            zero_count = np.sum(valid_density == 0.0)
            near_zero_count = np.sum((valid_density > 0.0) & (valid_density < 1e10))
            print(f"[Density statistics] Delay={delay_val:.6f}")
            print(f"  Valid points: {len(valid_density)} / {density_map.size}")
            print(f"  Zero-valued points: {zero_count}")
            print(f"  Range: [{np.min(valid_density):.3e}, {np.max(valid_density):.3e}] cm⁻³")
            print(f"  Mean: {np.mean(valid_density):.3e} cm⁻³")
        
        # Find row closest to requested y
        iy = int(np.argmin(np.abs(y_mm - float(y_profile_mm))))
        profile = density_map[iy, :]
        
        # Plot 1: Density profile at y_profile_mm
        fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 4.5), constrained_layout=True)
        ax1.plot(x_mm, profile, "-k", linewidth=1.8)
        ax1.set_xlabel("x (mm)  (radial)")
        ax1.set_ylabel("Number density (cm$^{-3}$)")
        ax1.set_title(f"Density profile at y={y_mm[iy]:.3f} mm\ndelay={1000 * delay_val:.3f} ms  |  gas={gas}")
        ax1.grid(True, alpha=0.3)
        
        if show:
            fig1.canvas.draw_idle()
        
        # Plot 2: Full 2D density map
        fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 5), constrained_layout=True)
        
        # Display scaling
        m = np.nanpercentile(np.abs(density_map), float(robust_pct))
        m = max(float(m), 1e-30)
        
        extent = [0.0, float(x_mm[-1]), 0.0, float(y_mm[-1])]
        im = ax2.imshow(
            density_map,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=m,
        )
        ax2.axhline(y=y_mm[iy], linestyle="--", linewidth=1.5, label=f"Profile at y={y_mm[iy]:.2f}mm")
        ax2.set_title(f"Density map — delay={1000*delay_val:.3f} ms")
        ax2.set_xlabel("x (mm)  (radial)")
        ax2.set_ylabel("y (mm)  (0 = nozzle edge)")
        cb = fig2.colorbar(im, ax=ax2, label="Density (cm$^{-3}$)")
        ax2.legend()
        
        if show:
            fig2.canvas.draw_idle()
            plt.show()
        
        return fig1, fig2, result


def phasemap_generator(folder_path = None):
    return PhaseMapGenerator(folder_path)

def slit_nozzle_analysis(perpendicular_dir = None, parallel_dir = None):
    return SlitNozzleAnalysis(perpendicular_dir, parallel_dir)

def cylindrical_nozzle_analysis(cylindrical_dir = None):
    return CylindricalNozzleAnalysis(cylindrical_dir)

def plot_density_from_saved(
        delay_folder: str | Path | None = None,
        *,
        robust_pct: float = 98.0,
        show: bool = True,
):
    """
    Plot a saved density map from a delay folder.
    
    Loads pre-computed density files (from compute_density_from_cylindrical_phase_map)
    and creates a 2D density map visualization with correct physical dimensions.
    
    Prompts user to select folder if not provided.
    
    Expected files in delay_folder:
        - density_map.npy: 2D density array in cm⁻³
        - x_mm.npy: x (radial) coordinates in mm
        - y_mm.npy: y (axial) coordinates in mm
        - meta.npy (optional): metadata dictionary
    
    Args:
        delay_folder: Path to delay folder (e.g., "Ar_35bar/delay_0.015000").
                     If None, prompts user to select.
        robust_pct: Percentile for display scaling
        show: Whether to call plt.show()
        
    Returns:
        fig, ax, density_map, x_mm, y_mm, title
    """
    # Prompt for folder if not provided
    if delay_folder is None or str(delay_folder).strip() == "":
        delay_folder = ask_for_folder(
            title="Select density folder (e.g., Ar_35bar/delay_0.015000)",
            initialdir=r"C:/Users/vmlab/Documents/density_measurements"
        )
    
    delay_folder = Path(delay_folder)
    
    if not delay_folder.is_dir():
        raise FileNotFoundError(f"Delay folder not found: {delay_folder}")
    
    # Load saved arrays
    density_path = delay_folder / "density_map.npy"
    x_path = delay_folder / "x_mm.npy"
    y_path = delay_folder / "y_mm.npy"
    meta_path = delay_folder / "meta.npy"
    
    if not density_path.exists():
        raise FileNotFoundError(f"density_map.npy not found in {delay_folder}")
    if not x_path.exists():
        raise FileNotFoundError(f"x_mm.npy not found in {delay_folder}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_mm.npy not found in {delay_folder}")
    
    # Load data
    density_map = np.load(density_path).astype(np.float64)
    x_mm = np.load(x_path).astype(np.float32)
    y_mm = np.load(y_path).astype(np.float32)
    
    # Load metadata if available
    meta = None
    if meta_path.exists():
        try:
            meta = np.load(meta_path, allow_pickle=True).item()
        except Exception:
            meta = None
    
    # Extract delay from folder name
    delay_str = delay_folder.name
    if delay_str.startswith("delay_"):
        try:
            delay_val = float(delay_str.split("_", 1)[1])
        except Exception:
            delay_val = None
    else:
        delay_val = None
    
    # Extract gas and metadata if available
    gas = None
    if meta:
        gas = meta.get("gas", None)
    
    # Get parent folder name for title (e.g., "Ar_35bar")
    parent_name = delay_folder.parent.name
    
    # Create title
    if gas:
        title = f"{parent_name}\n{delay_str}  |  {gas}"
    else:
        title = f"{parent_name}\n{delay_str}"
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), constrained_layout=True)
    
    # Display scaling
    m = np.nanpercentile(np.abs(density_map), float(robust_pct))
    m = max(float(m), 1e-30)
    
    extent = [0.0, float(x_mm[-1]), 0.0, float(y_mm[-1])]
    im = ax.imshow(
        density_map,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=m,
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)  (radial)")
    ax.set_ylabel("y (mm)  (0 = nozzle edge)")
    cb = fig.colorbar(im, ax=ax, label="Density (cm$^{-3}$)")
    
    # Add statistics text if available
    if meta and delay_val is not None:
        stats_text = f"Delay: {delay_val*1000:.3f} ms\n"
        if "width_mm" in meta:
            stats_text += f"Width: {meta['width_mm']:.1f} mm"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment="top", fontsize=9,
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    if show:
        plt.show()
    
    return fig, ax, density_map, x_mm, y_mm, title


_PROFILE_Y_RE = re.compile(r"_y(?P<y>[+-]?\d+(?:\.\d+)?)mm", re.IGNORECASE)

def extract_max_density_per_delay_from_run(run_dir: str | Path, profile_y_mm: Optional[float] = None) -> dict[float, float]:
    """
    Extract the maximum density from saved profile files under a single run folder.

    Parameters:
      run_dir: path to a run folder containing subfolders like 'delay_0.020000'
      profile_y_mm: if given, select the profile file whose embedded y (from filename
                    like 'profile_y2.000mm.npy') is closest to this value. If not given,
                    the function will expect exactly one profile file per delay (preserving
                    prior behavior) and will raise if multiple are present.

    Returns: dict mapping delay (float) -> max profile value (float)
    """

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    out: dict[float, float] = {}

    for delay_dir in run_dir.iterdir():
        if not delay_dir.is_dir():
            continue
        if not delay_dir.name.startswith("delay_"):
            continue

        try:
            delay = float(delay_dir.name.split("_", 1)[1])
        except Exception:
            # skip folders that don't follow naming convention
            continue

        # find profile files (prefer 'profile_y...' but accept 'profile_' fallback)
        prof_files = [
            p for p in delay_dir.iterdir()
            if p.is_file() and p.suffix == ".npy" and p.name.startswith("profile_")
        ]

        if not prof_files:
            continue

        # If user did not request a specific y, preserve original strict behavior
        if profile_y_mm is None:
            if len(prof_files) > 1:
                raise RuntimeError(f"More than one profile in {delay_dir}; pass profile_y_mm to select")
            chosen = prof_files[0]
        else:
            # Try to pick the profile whose filename embeds the y value closest to profile_y_mm
            best = None
            best_diff = float("inf")
            for p in prof_files:
                m = _PROFILE_Y_RE.search(p.name)
                if m:
                    try:
                        yval = float(m.group('y'))
                    except Exception:
                        continue
                    diff = abs(yval - float(profile_y_mm))
                    if diff < best_diff:
                        best_diff = diff
                        best = p

            if best is None:
                # try meta.npy's recorded y_profile_mm if available
                meta_path = delay_dir / "meta.npy"
                if meta_path.exists():
                    try:
                        meta = np.load(meta_path, allow_pickle=True).item()
                        meta_y = meta.get("y_profile_mm", None)
                        if meta_y is not None:
                            # find profile closest to meta_y
                            for p in prof_files:
                                m = _PROFILE_Y_RE.search(p.name)
                                if m:
                                    try:
                                        yval = float(m.group('y'))
                                    except Exception:
                                        continue
                                    diff = abs(yval - float(meta_y))
                                    if diff < best_diff:
                                        best_diff = diff
                                        best = p
                    except Exception:
                        pass

            if best is None:
                # final fallback: choose most recently modified profile file
                prof_files_sorted = sorted(prof_files, key=lambda p: p.stat().st_mtime, reverse=True)
                best = prof_files_sorted[0]

            chosen = best

        try:
            prof = np.asarray(np.load(chosen), dtype=np.float64)
        except Exception:
            # skip if loading fails
            continue

        max_val = float(np.nanmax(prof))
        if np.isfinite(max_val):
            out[delay] = max_val

    return out

def plot_avg_max_density_vs_delay_from_saved_profiles():
    parent = Path(ask_for_folder(
        title="Select parent folder containing RUN folders",
        initialdir=r"C:/Users/vmlab/Documents/density_maps",
    ))

    run_dirs = [p for p in parent.iterdir() if p.is_dir()]
    if not run_dirs:
        raise ValueError("No run folders found")

    all_runs = []
    for rd in run_dirs:
        d = extract_max_density_per_delay_from_run(rd)
        if d:
            all_runs.append(d)

    if not all_runs:
        raise ValueError("No valid data found in any run")

    # delays are guaranteed identical across runs
    delays = np.asarray(sorted(all_runs[0].keys()), dtype=float)
    delays_ms = 1000.0 * delays

    mat = np.full((len(all_runs), len(delays)), np.nan)
    for i, d in enumerate(all_runs):
        for j, t in enumerate(delays):
            mat[i, j] = d.get(t, np.nan)

    mean = np.nanmean(mat, axis=0)

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), constrained_layout=True)
    ax.errorbar(delays_ms, mean, yerr=np.nanstd(mat, axis=0), fmt="-o", color="tab:blue", markersize=6, label="mean ± std across runs")
    # ax.plot(delays_ms, mean, "-o", color="tab:blue", markersize=6, linestyle="none", label="mean across runs")
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("Max density at y≈2 mm  [cm$^{-3}$]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    return {
        "parent": str(parent),
        "run_dirs": [str(p) for p in run_dirs],
        "delays": np.asarray(delays),
        "matrix": mat,
        "mean": mean,
    }

def plot_profiles_from_saved_runs(
    root_dir: str | Path | None = None,
    *,
    delay_dir_name: str = "delay_0.015000",
    y_target_mm: float | None = None,
    show: bool = True,
    save_fig: str | Path | None = None,
    figsize=(8, 5),
    xlabel: str = "x (mm)",
    ylabel: str = "Number Density (cm$^{-3}$)",
    title: str | None = None,
):
    """
    Scan `root_dir` for run subfolders (e.g., 'Ar-5bar'), look for `delay_dir_name` in each,
    load `profile_y*.npy` and `x_mm.npy` (with sensible fallbacks) and plot all profiles on one
    matplotlib figure.

    Parameters:
      root_dir: parent folder containing run subdirectories.
      delay_dir_name: name of the delay folder to inspect inside each run folder.
      y_target_mm: if provided, only loads profiles whose filename-embedded y matches this value
                   (closest match per run). If None, loads all profile_y*.npy files found.
      show: whether to plt.show() the figure.
      save_fig: optional path to save the figure (PNG/PDF).
      figsize, xlabel, ylabel, title: plotting options.

    Returns:
      A list of dicts for each plotted profile with keys: {"run_name", "profile_path", "y_mm", "x", "profile"}.
    """
    # If no root_dir provided, ask the user for the folder interactively
    if root_dir is None or str(root_dir).strip() == "":
        root = Path(ask_for_folder(title="Select parent folder containing RUN folders", initialdir=r"C:/Users/vmlab/Documents/data/density_maps"))
    else:
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Root folder does not exist: {root}")

    plotted = []

    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise ValueError(f"No subfolders (runs) found in: {root}")

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    for run in runs:
        delay_dir = run / delay_dir_name
        if not delay_dir.exists() or not delay_dir.is_dir():
            # skip runs that don't have the requested delay folder
            continue

        # find profile files
        profiles = sorted(list(delay_dir.glob("profile_y*.npy")))
        if not profiles:
            # try any profile_*.npy as fallback
            profiles = sorted(list(delay_dir.glob("profile_*.npy")))
        if not profiles:
            # nothing to plot for this run
            continue

        # if a target y is given, choose the profile closest to that y
        chosen_profiles = []
        if y_target_mm is not None:
            # pick the profile file whose filename embeds y nearest to target
            best = None
            best_diff = float('inf')
            for p in profiles:
                m = _PROFILE_Y_RE.search(p.name)
                if m:
                    try:
                        yval = float(m.group('y'))
                    except Exception:
                        continue
                    diff = abs(yval - float(y_target_mm))
                    if diff < best_diff and diff < 0.01:
                        best_diff = diff
                        best = (p, yval)
            if best is not None:
                chosen_profiles = [best]
            else:
                # if no filename-embedded y found, skip this run
                chosen_profiles = []
        else:
            # load all profiles and try to extract y from filename
            for p in profiles:
                m = _PROFILE_Y_RE.search(p.name)
                yval = float(m.group('y')) if (m and m.group('y')) else None
                chosen_profiles.append((p, yval))

        if not chosen_profiles:
            continue

        # load x_mm if available else try to infer from meta or profile length
        x_path = delay_dir / "x_mm.npy"
        x_arr = None
        if x_path.exists():
            try:
                x_arr = np.load(x_path)
            except Exception:
                x_arr = None

        if x_arr is None:
            # try to use meta.npy to get perp_width_mm and profile length
            meta_path = delay_dir / "meta.npy"
            perp_width_mm = None
            if meta_path.exists():
                try:
                    meta = np.load(meta_path, allow_pickle=True).item()
                    perp_width_mm = float(meta.get("perp_width_mm", perp_width_mm))
                except Exception:
                    perp_width_mm = None

        for p, yval in chosen_profiles:
            try:
                prof = np.load(p)
            except Exception:
                continue

            if x_arr is not None:
                x = x_arr
            else:
                # fallback: infer x uniformly over perp_width_mm if available, otherwise use unit indices
                if perp_width_mm is not None:
                    x = np.linspace(0.0, float(perp_width_mm), num=prof.size, endpoint=False)
                else:
                    x = np.arange(prof.size, dtype=np.float32)

            # label should be only the run name (user requested)
            label = f"{run.name}"

            ax.plot(x, prof, label=label)

            plotted.append({
                "run_name": run.name,
                "profile_path": str(p),
                "y_mm": yval,
                "x": x,
                "profile": prof,
            })

    if not plotted:
        raise ValueError(f"No profiles found under {root} for delay '{delay_dir_name}'")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ttl = title or f"Gas Profile at {y_target_mm}mm Above Nozzle"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    if save_fig is not None:
        try:
            fig.savefig(str(save_fig), dpi=200)
        except Exception as e:
            print(f"Warning: failed to save figure to {save_fig}: {e}")

    if show:
        plt.show()

    return plotted


def extract_and_save_profile(
    folder: str | Path | None = None,
    y_mm: float | None = None,
):
    """
    Load a saved density map and extract a profile at a specific y value.
    
    Parameters:
      folder: path to the delay folder containing density_map.npy, x_mm.npy, y_mm.npy, meta.npy
      y_mm: desired y position in mm; if None, user is prompted
    
    Returns:
      tuple: (profile, x_mm, y_target, folder_path) or None if cancelled
    
    Saves the profile as profile_y{value}.npy in the folder.
    """

    # Ask for folder if not provided
    if folder is None:
        folder = ask_for_folder(
            title="Select density analysis folder",
            initialdir=r"C:/Users/vmlab/Documents/density_measurements"
        )

        # from utilities.actions import select_directory
        # folder = select_directory()
        # if folder is None:
        #     print("No folder selected")
        #     return None
    
    folder = Path(folder)
    
    # Check required files
    required_files = ["density_map.npy", "x_mm.npy", "y_mm.npy", "meta.npy"]
    for fname in required_files:
        fpath = folder / fname
        if not fpath.exists():
            print(f"Error: {fname} not found in {folder}")
            return None
    
    # Load the data
    try:
        density_map = np.load(folder / "density_map.npy")
        x_mm_arr = np.load(folder / "x_mm.npy")
        y_mm_arr = np.load(folder / "y_mm.npy")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
    
    # Ask for y value if not provided
    if y_mm is None:
        print(f"Available y range: {y_mm_arr.min():.3f} to {y_mm_arr.max():.3f} mm")
        try:
            y_mm = float(input("Enter desired y position (mm): "))
        except ValueError:
            print("Invalid input")
            return None
    
    # Find closest y index
    closest_idx = np.argmin(np.abs(y_mm_arr - y_mm))
    y_target = y_mm_arr[closest_idx]
    
    print(f"Requested y = {y_mm:.3f} mm, closest available = {y_target:.3f} mm")
    
    # Extract profile at that y
    profile = density_map[closest_idx, :]
    
    # Save profile
    filename = f"profile_y{y_target:.3f}mm.npy"
    filepath = folder / filename
    
    try:
        np.save(filepath, profile)
        print(f"Profile saved to {filepath}")
    except Exception as e:
        print(f"Error saving profile: {e}")
        return None
    
    # return profile, x_mm_arr, y_target, folder
