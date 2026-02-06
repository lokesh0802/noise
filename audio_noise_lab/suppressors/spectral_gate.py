"""
Spectral Gating Noise Suppression.

Implements STFT-based spectral gating where frequency bins below
a threshold (estimated from noise floor) are attenuated.

Theory:
-------
Spectral gating works by:
1. Computing Short-Time Fourier Transform (STFT) of the signal
2. Estimating the noise floor from quiet portions or statistics
3. Creating a gain mask that attenuates bins below threshold
4. Applying smoothing to prevent musical noise artifacts
5. Reconstructing signal via inverse STFT

This is a classic approach used in audio workstations and plugins.
Uses librosa for optimized STFT/ISTFT operations.
"""

import numpy as np
from typing import Tuple, Optional
import librosa


def _stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform using librosa.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D)
    n_fft : int
        FFT size
    hop_length : int
        Hop size between frames
    window : str
        Window function name
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Complex STFT matrix and window function
    """
    # Use librosa's optimized STFT
    stft_matrix = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
    )
    
    # Get window for compatibility
    win = librosa.filters.get_window(window, n_fft, fftbins=True)
    
    return stft_matrix, win


def _istft(
    stft_matrix: np.ndarray,
    hop_length: int = 512,
    window: np.ndarray = None,
    n_fft: int = 2048,
    original_length: int = None,
) -> np.ndarray:
    """
    Compute Inverse Short-Time Fourier Transform using librosa.
    
    Uses overlap-add method with proper normalization.
    
    Parameters
    ----------
    stft_matrix : np.ndarray
        Complex STFT matrix
    hop_length : int
        Hop size between frames
    window : np.ndarray
        Window function used in STFT (ignored, librosa infers from stft)
    n_fft : int
        FFT size
    original_length : int
        Original signal length for trimming
        
    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal
    """
    if window is None:
        window = np.hanning(n_fft)
    
    n_frames = stft_matrix.shape[1]
    expected_length = n_fft + hop_length * (n_frames - 1)
    
    # Initialize output and normalization arrays
    output = np.zeros(expected_length, dtype=np.float64)
    window_sum = np.zeros(expected_length, dtype=np.float64)
    
    # Overlap-add
    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        output[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    
    # Normalize by window sum (avoid division by zero)
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum
    
    # Remove padding
    pad_length = n_fft // 2
    output = output[pad_length:]
    
    if original_length is not None:
        output = output[:original_length]
    
    return output


def _estimate_noise_floor(
    magnitude: np.ndarray,
    percentile: float = 10.0,
    time_smooth: int = 5,
) -> np.ndarray:
    """
    Estimate noise floor from magnitude spectrogram.
    
    Uses percentile-based estimation across time frames,
    assuming noise is consistently present at low levels.
    
    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrogram (freq x time)
    percentile : float
        Percentile for noise floor estimation (default: 10)
    time_smooth : int
        Number of frames for temporal smoothing
        
    Returns
    -------
    np.ndarray
        Estimated noise floor per frequency bin
    """
    # Estimate noise as low percentile across time
    noise_floor = np.percentile(magnitude, percentile, axis=1)
    
    # Smooth across frequency
    from scipy.ndimage import uniform_filter1d
    noise_floor = uniform_filter1d(noise_floor, size=3)
    
    return noise_floor


def _compute_gain_mask(
    magnitude: np.ndarray,
    noise_floor: np.ndarray,
    threshold_db: float = -12.0,
    reduction_db: float = -24.0,
    attack_time: float = 0.01,
    release_time: float = 0.1,
    sample_rate: int = 16000,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute gain mask for spectral gating.
    
    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrogram
    noise_floor : np.ndarray
        Estimated noise floor per frequency bin
    threshold_db : float
        Threshold above noise floor to open gate (dB)
    reduction_db : float
        Amount of reduction when gate is closed (dB)
    attack_time : float
        Gate attack time in seconds
    release_time : float
        Gate release time in seconds
    sample_rate : int
        Audio sample rate
    hop_length : int
        STFT hop length
        
    Returns
    -------
    np.ndarray
        Gain mask (0 to 1)
    """
    n_freq, n_frames = magnitude.shape
    
    # Convert threshold to linear
    threshold_linear = noise_floor[:, np.newaxis] * (10 ** (threshold_db / 20))
    reduction_linear = 10 ** (reduction_db / 20)
    
    # Compute initial mask (hard gate)
    mask = np.where(magnitude > threshold_linear, 1.0, reduction_linear)
    
    # Smooth mask temporally to reduce musical noise
    # Compute smoothing coefficients based on attack/release time
    frame_time = hop_length / sample_rate
    attack_coef = np.exp(-frame_time / max(attack_time, 1e-6))
    release_coef = np.exp(-frame_time / max(release_time, 1e-6))
    
    # Apply temporal smoothing per frequency bin
    smoothed_mask = np.zeros_like(mask)
    smoothed_mask[:, 0] = mask[:, 0]
    
    for t in range(1, n_frames):
        # Determine if attacking (increasing) or releasing (decreasing)
        increasing = mask[:, t] > smoothed_mask[:, t - 1]
        
        # Apply appropriate smoothing coefficient
        coef = np.where(increasing, attack_coef, release_coef)
        smoothed_mask[:, t] = coef * smoothed_mask[:, t - 1] + (1 - coef) * mask[:, t]
    
    return smoothed_mask


def spectral_gate(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    threshold_db: float = -12.0,
    reduction_db: float = -24.0,
    noise_percentile: float = 10.0,
) -> np.ndarray:
    """
    Apply spectral gating noise suppression.
    
    This is a manual STFT-based implementation that estimates
    the noise floor and applies frequency-dependent gating.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono, float32/64)
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT size (default: 2048)
    hop_length : int
        Hop size (default: 512)
    threshold_db : float
        Threshold above noise floor in dB (default: -12)
    reduction_db : float
        Reduction amount when gated in dB (default: -24)
    noise_percentile : float
        Percentile for noise floor estimation (default: 10)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    original_length = len(audio)
    
    # Compute STFT
    stft_matrix, window = _stft(audio, n_fft, hop_length)
    
    # Get magnitude and phase
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Estimate noise floor
    noise_floor = _estimate_noise_floor(magnitude, noise_percentile)
    
    # Compute gain mask
    gain_mask = _compute_gain_mask(
        magnitude, noise_floor,
        threshold_db=threshold_db,
        reduction_db=reduction_db,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    
    # Apply mask
    magnitude_processed = magnitude * gain_mask
    
    # Reconstruct complex STFT
    stft_processed = magnitude_processed * np.exp(1j * phase)
    
    # Inverse STFT
    output = _istft(stft_processed, hop_length, window, n_fft, original_length)
    
    return output.astype(np.float32)


def spectral_gate_adaptive(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    threshold_db: float = -12.0,
    reduction_db: float = -20.0,
    adaptation_rate: float = 0.05,
) -> np.ndarray:
    """
    Apply adaptive spectral gating with time-varying noise estimation.
    
    Continuously updates noise estimate based on signal statistics,
    making it more robust to non-stationary noise.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT size
    hop_length : int
        Hop size
    threshold_db : float
        Gate threshold above noise floor (dB)
    reduction_db : float
        Reduction when gated (dB)
    adaptation_rate : float
        Rate of noise estimate adaptation (0 to 1)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    original_length = len(audio)
    
    # Compute STFT
    stft_matrix, window = _stft(audio, n_fft, hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    n_freq, n_frames = magnitude.shape
    
    # Initialize adaptive noise estimate from first few frames
    init_frames = min(10, n_frames // 4)
    noise_estimate = np.mean(magnitude[:, :init_frames], axis=1)
    
    # Process each frame adaptively
    magnitude_processed = np.zeros_like(magnitude)
    
    threshold_linear = 10 ** (threshold_db / 20)
    reduction_linear = 10 ** (reduction_db / 20)
    
    for t in range(n_frames):
        frame_mag = magnitude[:, t]
        
        # Compute threshold based on current noise estimate
        threshold = noise_estimate * threshold_linear
        
        # Determine which bins are likely noise
        is_noise = frame_mag < (noise_estimate * 2)
        
        # Update noise estimate (only from likely noise frames)
        if np.mean(is_noise) > 0.5:  # Mostly noise frame
            noise_estimate = (
                (1 - adaptation_rate) * noise_estimate +
                adaptation_rate * frame_mag
            )
        
        # Apply gating
        gain = np.where(frame_mag > threshold, 1.0, reduction_linear)
        
        # Soft knee transition
        transition_region = (frame_mag > noise_estimate * 0.5) & (frame_mag < threshold)
        if np.any(transition_region):
            ratio = frame_mag[transition_region] / threshold[transition_region]
            gain[transition_region] = reduction_linear + (1 - reduction_linear) * ratio
        
        magnitude_processed[:, t] = frame_mag * gain
    
    # Apply frequency smoothing to reduce artifacts
    from scipy.ndimage import uniform_filter
    magnitude_processed = uniform_filter(magnitude_processed, size=(3, 1))
    
    # Reconstruct
    stft_processed = magnitude_processed * np.exp(1j * phase)
    output = _istft(stft_processed, hop_length, window, n_fft, original_length)
    
    return output.astype(np.float32)
