from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from skimage.restoration import unwrap_phase

from src.data_analysis import start_analysis


def get_interferometry_images (folder_path, roi=None):
    FOLDER_PATH = Path(folder_path)
    current_analysis = start_analysis(FOLDER_PATH)
    image_dict = current_analysis.get_image_dictionary()

    for file_name, _ in image_dict.items():
        hdf5_path = Path(folder_path + '/' + file_name)
        parker_state = current_analysis.get_hdf5_group_data(hdf5_path, 'pulse_generators/qc_a/pulse_generator_channels/parker_valve_trigger')['pulse_state']
        if parker_state == 'on':
            exp_img = image_dict[file_name]
        elif parker_state == 'off':
            ref_img = image_dict[file_name]
        else:
            print('the parker valve state is ' + parker_state + '/n check whats going on...')

    if roi is None:
        ref_roi = ref_img[1000:2200, 700:4000]
        exp_roi = exp_img[1000:2200, 700:4000]
    else:
        ref_roi = ref_img[roi[0]:roi[1], roi[2]:roi[3]]
        exp_roi = exp_img[roi[0]:roi[1], roi[2]:roi[3]]

    return ref_img, exp_img, ref_roi, exp_roi


def get_fourier(img_roi):
    """
    to be used on both ROI
    """
    img_fourier = np.fft.fft2(img_roi - np.mean(img_roi))
    img_fourier_shifted = np.fft.fftshift(img_fourier)
    return img_fourier_shifted

def get_dominant_frequency(img_fourier_shifted, debug = False, min_dc_distance = 15):
    """
    to be used on ref ROI
    """
    img_magnitude_spectrum = np.abs(img_fourier_shifted)

    # plot the middle column of ref_magnitude_spectrum
    magnitude_spectrum = np.abs(img_fourier_shifted)
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2

    # Zero out DC and low-frequency region around the center
    y, x = np.ogrid[:rows, :cols]
    dc_mask = (y - crow) ** 2 + (x - ccol) ** 2 < min_dc_distance ** 2
    magnitude_spectrum[dc_mask] = 0
    flat_indices = np.argpartition(magnitude_spectrum.flatten(), -2)[-2:]
    peak_indices = np.array(np.unravel_index(flat_indices, magnitude_spectrum.shape)).T
    dominant_frequency = peak_indices[np.argmin(peak_indices[:, 0])]

    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
        plt.scatter(dominant_frequency[1], dominant_frequency[0], color='red', marker='x')
        plt.title('Magnitude Spectrum with Dominant Frequency')
        plt.xlabel('Frequency Component (u)')
        plt.ylabel('Frequency Component (v)')
        plt.show()
        print("dominant_frequency (v, u):", dominant_frequency)

    return dominant_frequency

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

def demodulate_image(filtered_fourier, dominant_frequency):
    # shift the dominant frequency to the center
    rows, cols = filtered_fourier.shape
    crow, ccol = rows // 2, cols // 2
    shift_y = crow - dominant_frequency[0]
    shift_x = ccol - dominant_frequency[1]
    shifted_fourier = np.roll(np.roll(filtered_fourier, shift_y, axis=0), shift_x, axis=1)
    demodulated_image = np.fft.ifft2(shifted_fourier)
    return demodulated_image

def get_delta_phase_wrapped (ref_fourier_shifted, exp_fourier_shifted, dominant_frequency, debug = False):
    dominant_frequency_mask = create_gaussian_cut_mask(ref_fourier_shifted.shape, dominant_frequency, radius = 20)
    ref_filtered_fourier = ref_fourier_shifted * dominant_frequency_mask
    exp_filtered_fourier = exp_fourier_shifted * dominant_frequency_mask
    ref_demodulated = demodulate_image(ref_filtered_fourier, dominant_frequency)
    exp_demodulated = demodulate_image(exp_filtered_fourier, dominant_frequency)

    ref_phase = np.angle(ref_demodulated)
    ref_phase = unwrap_phase(ref_phase)
    exp_phase = np.angle(exp_demodulated)
    exp_phase = unwrap_phase(exp_phase)
    delta_phase = ref_phase - exp_phase

    delta_phase_wrapped = np.angle(np.exp(1j * delta_phase))
    # cut 100 px at the edges of the phasemap
    delta_phase_wrapped = delta_phase_wrapped[100:-100, 100:-100]

    if debug:
        limit = np.nanmax(np.abs(delta_phase_wrapped))
        norm = TwoSlopeNorm(
            vmin=-limit,
            vcenter=0,
            vmax=limit
        )
        plt.figure()
        im = plt.imshow(
            delta_phase_wrapped,
            cmap="seismic",
            norm=norm,
            aspect="auto"
        )
        plt.colorbar(im, label = "Phase [rad]")
        plt.show()

    return delta_phase_wrapped