"""
Wiener Filter Noise Suppression.

Implements frequency-domain Wiener filtering for optimal
linear noise reduction.

Theory:
-------
The Wiener filter is the optimal linear filter for signal estimation
in the presence of additive noise. It minimizes the mean squared error
between the estimated and true signal.

For a noisy signal Y = S + N, the Wiener filter gain is:
    H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
         = SNR(f) / (1 + SNR(f))

When the signal and noise PSDs are unknown, we estimate them from
the noisy observation using statistical methods.

Key properties:
- Optimal in MSE sense for stationary signals
- Preserves phase information
- Can introduce musical noise if not smoothed
- Requires accurate noise PSD estimation

Uses librosa for optimized STFT/ISTFT operations.
"""

import numpy as np
from typing import Tuple, Optional
import librosa


def _compute_stft(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> Tuple[np.ndarray, int]:
    """
    Compute STFT using librosa for better performance.
    
    Returns
    -------
    Tuple[np.ndarray, int]
        Complex STFT matrix (freq x time) and number of frequency bins
    """
    # Use librosa's optimized STFT
    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
    )
    
    return stft_matrix, stft_matrix.shape[0]


def _compute_istft(
    stft_matrix: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    original_length: Optional[int] = None,
) -> np.ndarray:
    """Compute inverse STFT using librosa."""
    audio = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        center=True,
        length=original_length,
    )
    
    return audio


def _estimate_noise_psd(
    magnitude_squared: np.ndarray,
    method: str = "minimum_statistics",
    smoothing_factor: float = 0.98,
) -> np.ndarray:
    """
    Estimate noise Power Spectral Density.
    
    Parameters
    ----------
    magnitude_squared : np.ndarray
        Squared magnitude spectrogram (freq x time)
    method : str
        Estimation method: 'minimum_statistics' or 'percentile'
    smoothing_factor : float
        Temporal smoothing factor for tracking
        
    Returns
    -------
    np.ndarray
        Estimated noise PSD (freq x time)
    """
    n_freq, n_frames = magnitude_squared.shape
    noise_psd = np.zeros_like(magnitude_squared)
    
    if method == "minimum_statistics":
        # Minimum statistics method (Martin, 2001)
        # Track minimum of smoothed power over a sliding window
        
        # Smoothed power estimate
        smoothed_power = np.zeros_like(magnitude_squared)
        smoothed_power[:, 0] = magnitude_squared[:, 0]
        
        for t in range(1, n_frames):
            smoothed_power[:, t] = (
                smoothing_factor * smoothed_power[:, t - 1] +
                (1 - smoothing_factor) * magnitude_squared[:, t]
            )
        
        # Track minimum over sliding window
        window_size = min(50, n_frames // 4)
        
        for t in range(n_frames):
            start = max(0, t - window_size)
            end = min(n_frames, t + window_size + 1)
            noise_psd[:, t] = np.min(smoothed_power[:, start:end], axis=1)
        
        # Bias compensation (minimum statistics underestimates)
        noise_psd *= 1.5
        
    else:  # percentile method
        # Use low percentile as noise estimate
        noise_floor = np.percentile(magnitude_squared, 15, axis=1)
        noise_psd = np.tile(noise_floor[:, np.newaxis], (1, n_frames))
    
    return noise_psd


def _compute_wiener_gain(
    signal_psd: np.ndarray,
    noise_psd: np.ndarray,
    floor: float = 0.1,
) -> np.ndarray:
    """
    Compute Wiener filter gain.
    
    H(f) = max(1 - noise_psd/signal_psd, floor)
    
    Parameters
    ----------
    signal_psd : np.ndarray
        Signal power spectral density estimate
    noise_psd : np.ndarray
        Noise power spectral density estimate
    floor : float
        Minimum gain to prevent complete suppression
        
    Returns
    -------
    np.ndarray
        Wiener gain (0 to 1)
    """
    # Avoid division by zero
    signal_psd_safe = np.maximum(signal_psd, 1e-10)
    
    # Compute a priori SNR estimate
    snr = np.maximum(signal_psd_safe - noise_psd, 0) / (noise_psd + 1e-10)
    
    # Wiener gain: SNR / (1 + SNR)
    gain = snr / (1 + snr)
    
    # Apply floor
    gain = np.maximum(gain, floor)
    
    return gain


def wiener_filter(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    noise_estimation: str = "minimum_statistics",
    gain_floor: float = 0.1,
) -> np.ndarray:
    """
    Apply Wiener filter noise suppression.
    
    Estimates noise PSD from the signal and applies optimal
    linear Wiener filtering in the frequency domain.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT size (default: 2048)
    hop_length : int
        Hop size (default: 512)
    noise_estimation : str
        Method for noise PSD estimation
    gain_floor : float
        Minimum gain to prevent complete suppression
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    original_length = len(audio)
    
    # Compute STFT
    stft_matrix, n_freq = _compute_stft(audio, n_fft, hop_length)
    
    # Get magnitude and phase
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    magnitude_squared = magnitude ** 2
    
    # Estimate noise PSD
    noise_psd = _estimate_noise_psd(
        magnitude_squared,
        method=noise_estimation,
    )
    
    # Compute Wiener gain
    wiener_gain = _compute_wiener_gain(
        magnitude_squared,
        noise_psd,
        floor=gain_floor,
    )
    
    # Apply smoothing to reduce musical noise
    from scipy.ndimage import uniform_filter
    wiener_gain = uniform_filter(wiener_gain, size=(3, 5))
    
    # Apply gain
    magnitude_filtered = magnitude * wiener_gain
    
    # Reconstruct
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    
    # Inverse STFT
    output = _compute_istft(stft_filtered, n_fft, hop_length, original_length=original_length)
    
    return output.astype(np.float32)


def wiener_filter_adaptive(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    alpha: float = 0.98,
    beta: float = 0.8,
    gain_floor: float = 0.1,
) -> np.ndarray:
    """
    Apply adaptive Wiener filter with decision-directed SNR estimation.
    
    Uses the decision-directed approach (Ephraim & Malah, 1984) for
    improved SNR estimation and reduced musical noise.
    
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
    alpha : float
        Decision-directed smoothing factor (default: 0.98)
    beta : float
        Noise estimate smoothing factor (default: 0.8)
    gain_floor : float
        Minimum gain
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    original_length = len(audio)
    
    # Compute STFT
    stft_matrix, n_freq = _compute_stft(audio, n_fft, hop_length)
    n_freq, n_frames = stft_matrix.shape
    
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    magnitude_squared = magnitude ** 2
    
    # Initialize
    noise_psd = np.mean(magnitude_squared[:, :min(10, n_frames // 4)], axis=1)
    priori_snr = np.ones(n_freq)
    magnitude_filtered = np.zeros_like(magnitude)
    
    for t in range(n_frames):
        frame_power = magnitude_squared[:, t]
        
        # Posteriori SNR
        post_snr = frame_power / (noise_psd + 1e-10)
        
        # Decision-directed a priori SNR estimation
        if t == 0:
            priori_snr = np.maximum(post_snr - 1, 0)
        else:
            # Use previous frame's estimate
            prev_estimate = (magnitude_filtered[:, t - 1] ** 2) / (noise_psd + 1e-10)
            priori_snr = alpha * prev_estimate + (1 - alpha) * np.maximum(post_snr - 1, 0)
        
        # Wiener gain
        gain = priori_snr / (1 + priori_snr)
        gain = np.maximum(gain, gain_floor)
        
        # Apply gain
        magnitude_filtered[:, t] = magnitude[:, t] * gain
        
        # Update noise estimate during speech pauses
        # Simple VAD based on frame energy
        frame_energy = np.mean(frame_power)
        noise_energy = np.mean(noise_psd)
        
        if frame_energy < 2 * noise_energy:
            # Likely noise frame, update estimate
            noise_psd = beta * noise_psd + (1 - beta) * frame_power
    
    # Reconstruct
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    output = _compute_istft(stft_filtered, n_fft, hop_length, original_length=original_length)
    
    return output.astype(np.float32)


def parametric_wiener(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    mu: float = 1.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """
    Apply parametric Wiener filter with adjustable aggressiveness.
    
    Allows trading off noise reduction vs. signal distortion
    through parameters.
    
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
    mu : float
        Over-subtraction factor (>1 more aggressive)
    sigma : float
        Spectral floor factor
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    original_length = len(audio)
    
    # Compute STFT
    stft_matrix, _ = _compute_stft(audio, n_fft, hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    magnitude_squared = magnitude ** 2
    
    # Estimate noise
    noise_psd = _estimate_noise_psd(magnitude_squared, method="percentile")
    
    # Parametric subtraction
    subtracted = magnitude_squared - mu * noise_psd
    
    # Apply spectral floor
    floor = sigma * noise_psd
    subtracted = np.maximum(subtracted, floor)
    
    # Convert back to magnitude
    magnitude_filtered = np.sqrt(subtracted)
    
    # Reconstruct
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    output = _compute_istft(stft_filtered, n_fft, hop_length, original_length=original_length)
    
    return output.astype(np.float32)
