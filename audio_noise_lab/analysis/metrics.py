"""
Audio Quality Metrics.

Implements research-grade metrics for evaluating noise suppression
quality using proper DSP mathematics.

Metrics implemented:
1. SNR (Signal-to-Noise Ratio) - Global power ratio in dB
2. Segmental SNR - Frame-by-frame SNR averaged over time
3. RMSE (Root Mean Square Error) - Time-domain difference
4. Log Spectral Distance - Frequency-domain similarity

All metrics are computed with proper handling of edge cases,
numerical stability, and floating-point precision.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import signal as scipy_signal


@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    
    snr_db: float
    segmental_snr_db: float
    rmse: float
    log_spectral_distance: float
    processing_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "SNR (dB)": f"{self.snr_db:.2f}",
            "Segmental SNR (dB)": f"{self.segmental_snr_db:.2f}",
            "RMSE": f"{self.rmse:.6f}",
            "Log Spectral Distance": f"{self.log_spectral_distance:.4f}",
            "Processing Time (ms)": f"{self.processing_time_ms:.2f}",
        }


def compute_snr(
    original: np.ndarray,
    processed: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.
    
    SNR measures the ratio of signal power to noise power:
        SNR = 10 * log10(P_signal / P_noise)
    
    Here we treat 'original' as the noisy signal and compute
    how much noise was removed:
        SNR_improvement = 10 * log10(P_original / P_residual)
    
    where P_residual = P_original - P_processed represents
    the removed noise/signal.
    
    For noise reduction evaluation, higher SNR indicates
    more aggressive noise removal (which isn't always better).
    
    Parameters
    ----------
    original : np.ndarray
        Original (noisy) audio signal
    processed : np.ndarray
        Processed (denoised) audio signal
    eps : float
        Small value for numerical stability
        
    Returns
    -------
    float
        SNR in decibels
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute signal power (using processed as the "signal")
    signal_power = np.mean(processed ** 2)
    
    # Compute residual (what was removed)
    residual = original - processed
    noise_power = np.mean(residual ** 2)
    
    # Handle edge cases
    if signal_power < eps:
        return -np.inf  # No signal
    if noise_power < eps:
        return np.inf  # No noise removed (or identical signals)
    
    snr = 10 * np.log10(signal_power / (noise_power + eps))
    
    return float(snr)


def compute_segmental_snr(
    original: np.ndarray,
    processed: np.ndarray,
    frame_length: int = 256,
    hop_length: int = 128,
    eps: float = 1e-10,
    snr_range: Tuple[float, float] = (-10.0, 35.0),
) -> float:
    """
    Compute Segmental SNR in dB.
    
    Segmental SNR computes SNR for each frame and averages,
    which better correlates with perceived quality than global SNR.
    
    SNR_seg = (1/N) * sum(SNR_frame[i])
    
    where each frame SNR is clipped to a valid range to prevent
    extreme values from dominating the average.
    
    Parameters
    ----------
    original : np.ndarray
        Original (noisy) audio signal
    processed : np.ndarray
        Processed (denoised) audio signal
    frame_length : int
        Frame length in samples (default: 256 = 16ms at 16kHz)
    hop_length : int
        Hop between frames (default: 128)
    eps : float
        Small value for numerical stability
    snr_range : tuple
        (min, max) SNR values for clipping
        
    Returns
    -------
    float
        Average segmental SNR in decibels
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute number of frames
    n_frames = (min_len - frame_length) // hop_length + 1
    
    if n_frames < 1:
        return compute_snr(original, processed)  # Fall back to global SNR
    
    snr_values = []
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        orig_frame = original[start:end]
        proc_frame = processed[start:end]
        
        # Signal power (processed frame)
        signal_power = np.mean(proc_frame ** 2)
        
        # Noise power (residual)
        residual = orig_frame - proc_frame
        noise_power = np.mean(residual ** 2)
        
        # Skip silent frames
        if signal_power < eps:
            continue
        
        # Compute frame SNR
        if noise_power < eps:
            frame_snr = snr_range[1]  # Max value for very clean frames
        else:
            frame_snr = 10 * np.log10(signal_power / (noise_power + eps))
        
        # Clip to valid range
        frame_snr = np.clip(frame_snr, snr_range[0], snr_range[1])
        snr_values.append(frame_snr)
    
    if not snr_values:
        return 0.0
    
    return float(np.mean(snr_values))


def compute_rmse(
    original: np.ndarray,
    processed: np.ndarray,
) -> float:
    """
    Compute Root Mean Square Error between signals.
    
    RMSE measures the average magnitude of the difference:
        RMSE = sqrt(mean((original - processed)^2))
    
    Lower RMSE indicates the processed signal is closer to
    the original (less modification).
    
    For noise reduction, some RMSE is expected as noise is
    removed. Very low RMSE may indicate insufficient processing.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
        
    Returns
    -------
    float
        RMSE value (0 to ~1 for normalized audio)
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute RMSE
    mse = np.mean((original - processed) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def compute_log_spectral_distance(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    eps: float = 1e-10,
) -> float:
    """
    Compute Log Spectral Distance (LSD).
    
    LSD measures the difference between magnitude spectra in log domain:
        LSD = sqrt(mean((10*log10(S1) - 10*log10(S2))^2))
    
    where S1 and S2 are power spectra. This metric is commonly used
    in speech coding and enhancement evaluation.
    
    Lower LSD indicates more similar spectral characteristics.
    Typical values: 0-2 (excellent), 2-4 (good), >4 (poor).
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT size
    hop_length : int
        Hop length for STFT
    eps : float
        Small value for numerical stability
        
    Returns
    -------
    float
        Average Log Spectral Distance in dB
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute STFT for both signals
    _, _, orig_stft = scipy_signal.stft(
        original, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    _, _, proc_stft = scipy_signal.stft(
        processed, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    # Compute power spectra
    orig_power = np.abs(orig_stft) ** 2 + eps
    proc_power = np.abs(proc_stft) ** 2 + eps
    
    # Compute log spectral distance per frame
    log_diff = 10 * np.log10(orig_power) - 10 * np.log10(proc_power)
    
    # Average over frequency, then over time
    lsd_per_frame = np.sqrt(np.mean(log_diff ** 2, axis=0))
    lsd = np.mean(lsd_per_frame)
    
    return float(lsd)


def compute_spectral_convergence(
    original: np.ndarray,
    processed: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> float:
    """
    Compute Spectral Convergence metric.
    
    SC = ||S_orig - S_proc||_F / ||S_orig||_F
    
    where ||.||_F is the Frobenius norm. Values close to 0
    indicate similar spectra.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
    n_fft : int
        FFT size
    hop_length : int
        Hop length
        
    Returns
    -------
    float
        Spectral convergence (0 = identical)
    """
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute magnitude spectra
    _, _, orig_stft = scipy_signal.stft(
        original, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    _, _, proc_stft = scipy_signal.stft(
        processed, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    orig_mag = np.abs(orig_stft)
    proc_mag = np.abs(proc_stft)
    
    # Compute Frobenius norms
    diff_norm = np.linalg.norm(orig_mag - proc_mag, 'fro')
    orig_norm = np.linalg.norm(orig_mag, 'fro')
    
    if orig_norm < 1e-10:
        return 0.0
    
    return float(diff_norm / orig_norm)


def compute_pesq_if_available(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    """
    Compute PESQ score if the library is available.
    
    PESQ (Perceptual Evaluation of Speech Quality) is the
    ITU-T standard for speech quality assessment.
    
    Parameters
    ----------
    original : np.ndarray
        Reference (clean) signal
    processed : np.ndarray
        Degraded signal
    sample_rate : int
        Sample rate (8000 or 16000 Hz)
        
    Returns
    -------
    float or None
        PESQ score (-0.5 to 4.5) or None if unavailable
    """
    try:
        from pesq import pesq
        
        # PESQ requires 8kHz or 16kHz
        if sample_rate not in [8000, 16000]:
            return None
        
        mode = "wb" if sample_rate == 16000 else "nb"
        
        min_len = min(len(original), len(processed))
        score = pesq(sample_rate, original[:min_len], processed[:min_len], mode)
        return float(score)
        
    except ImportError:
        return None
    except Exception:
        return None


def compute_all_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int = 16000,
    processing_time_ms: float = 0.0,
) -> MetricsResult:
    """
    Compute all metrics for a processed audio signal.
    
    Parameters
    ----------
    original : np.ndarray
        Original (noisy) audio signal
    processed : np.ndarray
        Processed (denoised) audio signal
    sample_rate : int
        Sample rate in Hz
    processing_time_ms : float
        Processing time in milliseconds
        
    Returns
    -------
    MetricsResult
        All computed metrics in a structured format
    """
    # Ensure same length and matching dtypes
    min_len = min(len(original), len(processed))
    original = original[:min_len].astype(np.float64)
    processed = processed[:min_len].astype(np.float64)
    
    # Compute all metrics
    snr = compute_snr(original, processed)
    seg_snr = compute_segmental_snr(original, processed)
    rmse = compute_rmse(original, processed)
    lsd = compute_log_spectral_distance(original, processed, sample_rate)
    
    return MetricsResult(
        snr_db=snr,
        segmental_snr_db=seg_snr,
        rmse=rmse,
        log_spectral_distance=lsd,
        processing_time_ms=processing_time_ms,
    )


def interpret_metrics(metrics: MetricsResult) -> dict:
    """
    Provide interpretation of metric values.
    
    Parameters
    ----------
    metrics : MetricsResult
        Computed metrics
        
    Returns
    -------
    dict
        Interpretations for each metric
    """
    interpretations = {}
    
    # SNR interpretation
    if metrics.snr_db > 20:
        interpretations["snr"] = "Significant noise removal"
    elif metrics.snr_db > 10:
        interpretations["snr"] = "Moderate noise removal"
    elif metrics.snr_db > 0:
        interpretations["snr"] = "Light noise removal"
    else:
        interpretations["snr"] = "Minimal processing or noise amplification"
    
    # LSD interpretation
    if metrics.log_spectral_distance < 1:
        interpretations["lsd"] = "Excellent spectral preservation"
    elif metrics.log_spectral_distance < 2:
        interpretations["lsd"] = "Good spectral preservation"
    elif metrics.log_spectral_distance < 4:
        interpretations["lsd"] = "Moderate spectral change"
    else:
        interpretations["lsd"] = "Significant spectral distortion"
    
    # RMSE interpretation
    if metrics.rmse < 0.01:
        interpretations["rmse"] = "Minimal signal change"
    elif metrics.rmse < 0.05:
        interpretations["rmse"] = "Light modification"
    elif metrics.rmse < 0.1:
        interpretations["rmse"] = "Moderate modification"
    else:
        interpretations["rmse"] = "Heavy modification"
    
    # Processing time
    if metrics.processing_time_ms < 100:
        interpretations["time"] = "Real-time capable"
    elif metrics.processing_time_ms < 1000:
        interpretations["time"] = "Near real-time"
    else:
        interpretations["time"] = "Offline processing"
    
    return interpretations
