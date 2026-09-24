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

from src.data_analysis import start_analysis
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any, Union
import numpy.typing as npt
import re
from pathlib import Path
from tkinter import Tk, filedialog
import config

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

# Helper function for all classes
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
                initialdir=config.RAW_FILES_DIR
            )
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
        self.analysis = start_analysis(folder_path)


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

    def get_interferometry_phase(
            self,
            file_pair,
            expansion_mm: float = 3.0,
            nozzle_width_mm: float = None,
            px_per_mm: float = None
    ):
        """
        Compute the wrapped phase difference map.
        Requires EITHER nozzle_width_mm OR px_per_mm to calculate physical scaling.
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

        # --- Dynamic Scaling Logic ---
        if px_per_mm is not None:
            # User provided exact pixel scaling
            actual_px_per_mm = float(px_per_mm)
            actual_nozzle_width_mm = nozzle_width_px / actual_px_per_mm
        elif nozzle_width_mm is not None:
            # User provided physical nozzle width
            actual_nozzle_width_mm = float(nozzle_width_mm)
            actual_px_per_mm = nozzle_width_px / actual_nozzle_width_mm
        else:
            raise ValueError("You must provide either 'px_per_mm' or 'nozzle_width_mm' for scaling.")

        expansion_px = int(round(expansion_mm * actual_px_per_mm))

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
