"""
Noisereduce Library Wrapper.

Provides a clean interface to the noisereduce library for both
stationary and non-stationary noise reduction.

The noisereduce library implements spectral gating with sophisticated
noise profiling and is widely used in audio processing applications.

Theory:
-------
The noisereduce library uses spectral gating similar to our manual
implementation but with additional features:

Stationary mode:
- Estimates noise from a noise clip or the signal itself
- Assumes noise characteristics don't change over time
- Better for constant background noise (fans, hum)

Non-stationary mode:
- Adapts to time-varying noise
- Uses running statistics for noise estimation
- Better for variable noise (traffic, crowd)

Reference: https://github.com/timsainb/noisereduce
"""

import numpy as np
from typing import Optional


def _check_noisereduce_available() -> bool:
    """Check if noisereduce library is available."""
    try:
        import noisereduce
        return True
    except ImportError:
        return False


def noisereduce_stationary(
    audio: np.ndarray,
    sample_rate: int,
    prop_decrease: float = 0.8,
    n_fft: int = 2048,
    hop_length: int = 512,
    noise_clip: Optional[np.ndarray] = None,
    use_tqdm: bool = False,
) -> np.ndarray:
    """
    Apply noisereduce library with stationary noise assumption.
    
    Estimates noise profile from the signal (or provided clip)
    and applies spectral gating with smoothing.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    prop_decrease : float
        Proportion to reduce noise by (0 to 1, default: 0.8)
    n_fft : int
        FFT size (default: 2048)
    hop_length : int
        Hop size (default: 512)
    noise_clip : np.ndarray, optional
        Separate noise sample for profiling
    use_tqdm : bool
        Show progress bar (default: False)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
        
    Raises
    ------
    ImportError
        If noisereduce library is not installed
    """
    if not _check_noisereduce_available():
        raise ImportError(
            "noisereduce library not installed. "
            "Install with: pip install noisereduce"
        )
    
    import noisereduce as nr
    
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Apply stationary noise reduction
    output = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=True,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        hop_length=hop_length,
        y_noise=noise_clip,
        use_tqdm=use_tqdm,
    )
    
    return output.astype(np.float32)


def noisereduce_nonstationary(
    audio: np.ndarray,
    sample_rate: int,
    prop_decrease: float = 0.8,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_constant_s: float = 2.0,
    freq_mask_smooth_hz: int = 500,
    time_mask_smooth_ms: int = 50,
    use_tqdm: bool = False,
) -> np.ndarray:
    """
    Apply noisereduce library with non-stationary noise handling.
    
    Uses adaptive noise estimation for time-varying noise sources.
    Better for variable background noise like traffic or crowds.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    prop_decrease : float
        Proportion to reduce noise by (0 to 1, default: 0.8)
    n_fft : int
        FFT size (default: 2048)
    hop_length : int
        Hop size (default: 512)
    time_constant_s : float
        Time constant for noise estimation (default: 2.0s)
    freq_mask_smooth_hz : int
        Frequency smoothing width in Hz (default: 500)
    time_mask_smooth_ms : int
        Temporal smoothing width in ms (default: 50)
    use_tqdm : bool
        Show progress bar (default: False)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
        
    Raises
    ------
    ImportError
        If noisereduce library is not installed
    """
    if not _check_noisereduce_available():
        raise ImportError(
            "noisereduce library not installed. "
            "Install with: pip install noisereduce"
        )
    
    import noisereduce as nr
    
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Apply non-stationary noise reduction
    output = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=False,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        hop_length=hop_length,
        time_constant_s=time_constant_s,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        use_tqdm=use_tqdm,
    )
    
    return output.astype(np.float32)


def noisereduce_with_profile(
    audio: np.ndarray,
    sample_rate: int,
    noise_start_ms: int = 0,
    noise_end_ms: int = 500,
    prop_decrease: float = 0.9,
) -> np.ndarray:
    """
    Apply noisereduce using a noise profile from the audio itself.
    
    Extracts a noise sample from the specified time range and uses
    it for noise profiling. Useful when the recording has a known
    noise-only section (e.g., silence at the beginning).
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    noise_start_ms : int
        Start of noise sample in milliseconds
    noise_end_ms : int
        End of noise sample in milliseconds
    prop_decrease : float
        Proportion to reduce noise (default: 0.9)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
    """
    if not _check_noisereduce_available():
        raise ImportError(
            "noisereduce library not installed. "
            "Install with: pip install noisereduce"
        )
    
    import noisereduce as nr
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract noise sample
    start_sample = int(noise_start_ms * sample_rate / 1000)
    end_sample = int(noise_end_ms * sample_rate / 1000)
    
    # Ensure valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    if end_sample <= start_sample:
        # Fall back to automatic noise estimation
        noise_clip = None
    else:
        noise_clip = audio[start_sample:end_sample]
    
    output = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        y_noise=noise_clip,
        prop_decrease=prop_decrease,
        stationary=True,
    )
    
    return output.astype(np.float32)


def is_noisereduce_available() -> bool:
    """
    Check if noisereduce library is available.
    
    Returns
    -------
    bool
        True if noisereduce is installed
    """
    return _check_noisereduce_available()
