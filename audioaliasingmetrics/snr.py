import numpy as np
from numba import njit
from typing import Union

from .utils import *


@njit
def inner_snr(
    periodigram: np.ndarray[np.float32],
    samplerate: float,
    fundamental: float,
    harmonics: np.ndarray[np.int32],
    harmonics_excluded: bool,
) -> float:
    # Find all the bins of the fundamental
    fund_bin = find_peak_bin_from_freq(fundamental, periodigram, samplerate)
    fund_peak_bins = find_peak_bins(fund_bin, periodigram)

    # Find all the bins of the harmonics
    harmonics_bins = np.empty((0), dtype=np.int32)
    for harmonic_factor in harmonics:
        harmonic = harmonic_factor * fundamental
        harmonic_bin = find_peak_bin_from_freq(harmonic, periodigram, samplerate)
        harmonics_bins = np.append(
            harmonics_bins, find_peak_bins(harmonic_bin, periodigram)
        )

    # Compute the signal power
    if harmonics_excluded:
        signal_bins = fund_peak_bins
    else:
        signal_bins = np.concatenate((fund_peak_bins, harmonics_bins))
    signal_power = np.sum(periodigram[signal_bins])

    # Compute the noise power
    to_delete = np.concatenate((fund_peak_bins, harmonics_bins))
    noise = np.delete(periodigram, to_delete)
    noise_power = np.sum(noise[1:])  # Exclude DC

    return 10 * np.log10(signal_power / noise_power)


def snr(
    magnitude: np.ndarray[np.float32],
    samplerate: float,
    fundamental: float,
    harmonics: Union[Harmonics, np.ndarray[np.int32]] = list(),
    harmonics_excluded: bool = False,
) -> float:
    """Compute the signal-to-noise ratio (SNR) in decibels of a signal,
    using its spectral magnitude, the fundamental of the signal and its harmonics

    Args:
        magnitude (np.ndarray[np.float32]): A 1D array of spectral magnitude of the noisy signal
        samplerate (float): Samplerate
        fundamental (float): Fundamental frequency of the signal. The lower the frequency, the most accurate it needs to be.
            It is used to estimate the frequencies of the harmonics
        harmonics (Union[Harmonics, np.ndarray[np.int32]], optional): Specify which harmonics are part of the signal.
            Either a Harmonic enum (ALL / ODD / EVEN)  or an array/list of >=2 integers
            (eg : [2, 4, 6] represents the first 3 even harmonics). Defaults to an empty array.
        harmonics_excluded (bool, optional): Should the harmonics be excluded from the signal power computation. This matches
            the Matlab SNR method where it expects the input to be a sinusoïdal signal. Even if excluded from the signal power, harmonics will
            always be excluded from the noise power. Defaults to False.


    Returns:
        float: An estimation of the SNR in dB
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

    return inner_snr(
        np.square(magnitude), samplerate, fundamental, harmonics, harmonics_excluded
    )
