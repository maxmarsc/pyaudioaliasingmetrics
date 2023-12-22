import numpy as np
from numba import njit
from math import floor, ceil
from enum import Enum


class Harmonics(Enum):
    """Enumeration describing the expected harmonics of the signal.
    Possible values are : ALL, ODD, EVEN
    """

    ALL = 0
    ODD = 1
    EVEN = 2

    def compute_harmonics(
        self, fundamental: float, samplerate: float
    ) -> np.ndarray[np.int32]:
        nyquist = samplerate / 2.0
        if self == Harmonics.ALL:
            return np.arange(2, floor(nyquist / fundamental) + 1)
        elif self == Harmonics.ODD:
            return np.arange(3, floor(nyquist / fundamental) + 1, step=2)
        else:
            return np.arange(2, floor(nyquist / fundamental) + 1, step=2)


# def interpolate_fundamental(ps: np.ndarray[float], sr: float) -> float:
#     fund_bin = np.argmax(ps[1:]) + 1  # Exclude DC
#     corresponding_freq = fund_bin / (ps.shape[0] - 1) * (sr / 2)
#     if ps[fund_bin - 1] > ps[fund_bin + 1]:
#         bin_range = np.arange(fund_bin - 2, fund_bin + 2)
#         # points = ps[fund_bin - 2 : fund_bin + 2]
#     else:
#         # points = ps[fund_bin - 1 : fund_bin + 3]
#         bin_range = np.arange(fund_bin - 1, fund_bin + 3)
#     points = ps[bin_range]
#     freqs = bin_range / (ps.shape[0] - 1) * (sr / 2)

#     poly = interpolate.lagrange(freqs, points)

#     # Convert to a form that can be differentiated
#     coefs = Polynomial(poly).coef
#     deriv = np.polyder(coefs)

#     # Find roots of the derivative and select the one that's closest to the original peak
#     roots = np.roots(deriv)
#     real_roots = roots[np.isreal(roots)].real
#     closest_peak = real_roots[np.argmin(np.abs(real_roots - corresponding_freq))]

#     return closest_peak


@njit
def find_peak_bin_from_freq(
    freq_hint: float, ps: np.ndarray[float], sr: float, search_width_hz=10
) -> int:
    """Find the index of the bin of the highest point of a frequency peak

    Args:
        freq_hint (float): The expected frequency of the peak
        ps (np.ndarray[float]): The periodigram
        sr (float): samplerate
        search_width_hz (int, optional): Width of the search area for the peak. Defaults to 10.

    Returns:
        int: The index of the bin
    """
    num_bins = ps.shape[0]
    hint_idx = floor(2.0 * freq_hint / sr * (num_bins - 1))
    search_width = ceil(search_width_hz / (sr / 2.0) * (num_bins - 1))
    left_limit = hint_idx - search_width
    search_center = search_width
    if left_limit < 0:
        search_center += left_limit
        left_limit = 0
    right_limit = min(num_bins, hint_idx + search_width + 1)
    search_slice = ps[left_limit:right_limit]

    return (
        find_nearest_peak_around(search_slice, search_center, search_width)
        + hint_idx
        - search_width
    )


@njit
def is_peak(slice_of_3: np.ndarray[float]) -> bool:
    return slice_of_3[1] > slice_of_3[0] and slice_of_3[1] > slice_of_3[2]


@njit
def find_nearest_peak_around(
    ps_slice: np.ndarray[float], center: int, search_width: int
) -> int:
    """Find the index of the bin of the nearest peak to the center

    Args:
        ps_slice (np.ndarray[float]): The slice in which to search
        center (int): The index of the center of the search
        search_width (int): The width of the search

    Returns:
        int: The index of the bin
    """
    center = ps_slice.shape[0] // 2
    for i in range(0, search_width):
        # right side first
        right_idx = center + i
        if right_idx + 1 < ps_slice.shape[0] and is_peak(
            ps_slice[right_idx - 1 : right_idx + 2]
        ):
            return right_idx

        # left side then
        left_idx = center - i
        if left_idx - 1 >= 0 and is_peak(ps_slice[left_idx - 1 : left_idx + 2]):
            return left_idx

    # fallback
    return np.argmax(ps_slice)


@njit
def find_peak_bins(
    peak_idx: int, ps: np.ndarray[float], search_width=1000
) -> np.ndarray[np.int32]:
    """Given the index of the bin of a peak, this will find all the bins of the peak

    Args:
        peak_idx (int): the index of the highest point of the peak
        ps (np.ndarray[float]): the periodigram
        search_width (int, optional): The width (in number of bins) for the search. Defaults to 1000.

    Returns:
        np.ndarray[int]: An array of all the bins of the peak, ordered
    """
    peak_bins = []
    peak = ps[peak_idx]

    # Find values before peak
    crt_val = peak
    for i in range(search_width):
        new_bin = peak_idx - i
        if new_bin < 0:
            break
        new_val = ps[new_bin]
        if new_val > crt_val:
            break
        peak_bins.append(new_bin)
        crt_val = new_val

    # Find values after peak
    crt_val = peak
    for i in range(search_width):
        new_bin = peak_idx + i
        if new_bin >= ps.shape[0]:
            break
        new_val = ps[new_bin]
        if new_val > crt_val:
            break
        peak_bins.append(new_bin)
        crt_val = new_val

    return np.array(peak_bins, dtype=np.int32)
