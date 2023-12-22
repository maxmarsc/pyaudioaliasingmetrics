# pyaudioaliasingmetrics
A Python implementation of SNR and SINAD metrics, using Numpy and Numba

These implementations expects three types of inputs :
- spectral magnitudes of the required signals
- the fundamental frequency of the signal
- a description of which harmonics are part of the signal

## Installation
You can install this library using pip
```shell
pip install pyaudioaliasingmetrics
```

## Usage
*Matlab suggest using a Kaiser window with a beta factor of 38*

### SNR
Computing the SNR of a saw wave at 440Hz, sampled at 44100Hz
```py
import numpy as np
import audioaliasingmetrics as aam

noisy_signal : np.ndarray[np.float32] = ... # audio time serie of the noisy signal

length = noisy_signal.shape[0]
window = window = np.kaiser(length, 38)
magnitude = np.abs(np.fft.rfft(noisy_signal * window))
snr = aam.snr(magnitude, 44100, 440.0, aam.Harmonics.ALL)
```

### SINAD
Computing the SINAD of a saw wave at 440Hz, sampled at 44100Hz
```py
import numpy as np
import audioaliasingmetrics as aam

noisy_signal : np.ndarray[np.float32] = ... # audio time serie of the noisy signal
clean_signal : np.ndarray[np.float32] = ... # audio time serie of the clean signal

length = noisy_signal.shape[0]
window = window = np.kaiser(length, 38)
noisy_magnitude = np.abs(np.fft.rfft(noisy_signal * window))
clean_magnitude = np.abs(np.fft.rfft(clean_signal * window))

sinad = aam.sinad(noisy_magnitude, clean_magnitude, 44100, 440.0, aam.Harmonics.ALL)
```

### Harmonics selection
You can select which harmonics are part of your signal with the `harmonics` parameter of both methods. It supports two modes:

- A `Harmonics` enum value (`ODD` / `EVEN` / `ALL`) to respectively select all possible odd, even or both harmonics.
- A list/numpy array of integers values >2 of harmonics factors by which to multiply the fundamental with, to precisely select harmonics.