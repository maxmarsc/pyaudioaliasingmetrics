import numpy as np
from numba import njit
from typing import Union

from .utils import *


@njit
def inner_sinad(
    noisy_magnitudes: np.ndarray[np.float32],
    clean_magnitudes: np.ndarray[np.float32],
    samplerate: float,
    fundamental: float,
    harmonics: np.ndarray[np.int32],
) -> float:
    # Normalize according to the magnitude of the fundamental :
    fund_bin = find_peak_bin_from_freq(fundamental, noisy_magnitudes, samplerate)
    fund_peak_bins = find_peak_bins(fund_bin, noisy_magnitudes)

    mag_sum_noised = np.sum(noisy_magnitudes[fund_peak_bins[0] : fund_peak_bins[-1]])
    mag_sum_clean = np.sum(clean_magnitudes[fund_peak_bins[0] : fund_peak_bins[-1]])
    mag_ratio = mag_sum_noised / mag_sum_clean
    noisy_magnitudes *= mag_ratio
    noise_magnitudes = noisy_magnitudes - clean_magnitudes

    # Compute the PSD after normalization
    psd_clean = np.square(clean_magnitudes)
    psd_noise = np.square(noise_magnitudes)
    psd_noisy = np.square(noisy_magnitudes)

    # Find all the bins of the harmonics
    harmonics_bins = np.empty((0), dtype=np.int32)
    for harmonic_factor in harmonics:
        harmonic = harmonic_factor * fundamental
        harmonic_bin = find_peak_bin_from_freq(harmonic, psd_clean, samplerate)
        harmonics_bins = np.append(
            harmonics_bins, find_peak_bins(harmonic_bin, psd_noisy)
        )

    # Compute the signal power
    signal_bins = np.concatenate((fund_peak_bins, harmonics_bins))
    signal_power = np.sum(psd_clean[signal_bins])

    # Compute the noise & distortion power
    nad = np.delete(psd_noise, fund_peak_bins)
    nad_power = np.sum(nad[1:])

    return 10 * np.log10((signal_power + nad_power) / nad_power)


def sinad(
    noisy_magnitudes: np.ndarray[np.float32],
    clean_magnitudes: np.ndarray[np.float32],
    samplerate: float,
    fundamental: float,
    harmonics: Union[Harmonics, np.ndarray[np.int32]] = list(),
) -> float:
    """Compute the signal-to-noise and distortion ratio (SINAD) in decibels of a noisy signal,
    using the spectral magnitude of the noisy signal, the spectral magnitude of a clean version of the signal,
    the fundamental of the signal and its harmonics

    Args:
        noisy_magnitudes (np.ndarray[np.float32]): A 1D array of spectral magnitude of the noisy signal
        clean_magnitudes (np.ndarray[np.float32]): A 1D array of spectral magnitude of the clean signal
        samplerate (float): Samplerate
        fundamental (float): Fundamental frequency of the signal. The lower the frequency, the most accurate it needs to be.
            It is used to estimate the frequencies of the harmonics
        harmonics (Union[Harmonics, np.ndarray[np.int32]], optional): Specify which harmonics are part of the signal.
            Either a Harmonic enum (ALL / ODD / EVEN)  or an array/list of >=2 integers
            (eg : [2, 4, 6] represents the first 3 even harmonics). Defaults to an empty array.


    Returns:
        float: An estimation of the SINAD in dB
    """
    if samplerate <= 0.0:
        raise ValueError("Invalid samplerate value")

    if fundamental <= 0.0 or fundamental > samplerate / 2:
        raise ValueError("Invalid fundamental value")

    # Build the harmonic vector
    if not isinstance(harmonics, Harmonics):
        # Convert to numpy if needed
        if isinstance(harmonics, list):
            harmonics = np.array(harmonics, dtype=np.int32)

        # Sort if needed
        if harmonics.shape[0] > 0:
            harmonics = np.sort(harmonics)
            if harmonics[0] < 2:
                raise ValueError("Harmonics values must be >=2")
            top_harmonic = floor((samplerate / 2) / fundamental)
            if harmonics[-1] > top_harmonic:
                raise ValueError(
                    "Given your fundamental, the highest harmonic value should be {}".format(
                        top_harmonic
                    )
                )
    else:
        harmonics = harmonics.compute_harmonics(fundamental, samplerate)

    return inner_sinad(
        noisy_magnitudes, clean_magnitudes, samplerate, fundamental, harmonics
    )
