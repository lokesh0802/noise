"""
Bandpass and Frequency Filter Noise Suppression.

Implements IIR and FIR filters using SciPy for removing
out-of-band noise components.

Theory:
-------
Frequency filtering removes noise by attenuating signal components
outside a desired frequency range:

- Highpass: Removes low-frequency noise (rumble, hum, DC offset)
- Lowpass: Removes high-frequency noise (hiss, aliasing)
- Bandpass: Combines both, passing only a specific frequency band

For speech, the voice band (300-3400 Hz) contains most intelligibility.
A bandpass filter in this range removes:
- Low-freq: HVAC rumble, wind noise, handling noise
- High-freq: Electronic hiss, high-frequency interference

Filter design considerations:
- Order: Higher = sharper cutoff but more phase distortion
- Type: Butterworth (flat passband), Chebyshev (sharper cutoff)
- Zero-phase: Forward-backward filtering eliminates phase distortion
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


def _design_butterworth_filter(
    cutoff: float,
    sample_rate: int,
    filter_type: str = "low",
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth IIR filter using SOS (second-order sections).
    
    SOS representation provides better numerical stability than
    transfer function (ba) coefficients, especially for high-order filters.
    
    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz (or tuple for bandpass)
    sample_rate : int
        Sample rate in Hz
    filter_type : str
        Filter type: 'low', 'high', 'band', 'bandstop'
    order : int
        Filter order
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        SOS array for use with sosfiltfilt
    """
    nyquist = sample_rate / 2
    
    if isinstance(cutoff, (list, tuple)):
        normalized_cutoff = [f / nyquist for f in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if isinstance(normalized_cutoff, list):
        normalized_cutoff = [max(0.001, min(0.999, f)) for f in normalized_cutoff]
    else:
        normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
    
    # Use SOS (second-order sections) for better numerical stability
    sos = signal.butter(order, normalized_cutoff, btype=filter_type, output='sos')
    
    return sos, None  # Return None for 'a' to maintain interface


def _apply_filter_zerophase(
    audio: np.ndarray,
    sos: np.ndarray,
    a: np.ndarray = None,  # Kept for compatibility but unused
) -> np.ndarray:
    """
    Apply filter with zero phase distortion using sosfiltfilt.
    
    Uses second-order sections for better numerical stability
    than traditional ba coefficients.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sos : np.ndarray
        Second-order sections representation of filter
    a : np.ndarray
        Unused (kept for interface compatibility)
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    # Pad signal to reduce edge effects
    pad_length = min(len(audio) // 4, 1000)  # Adaptive padding
    
    if len(audio) <= 2 * pad_length:
        # Signal too short, use minimal padding
        return signal.sosfiltfilt(sos, audio)
    
    return signal.sosfiltfilt(sos, audio, padtype='odd', padlen=pad_length)


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply bandpass filter for voice-band isolation.
    
    Passes frequencies between low_cutoff and high_cutoff,
    attenuating everything else. Default values are optimized
    for speech intelligibility (telephone band).
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    low_cutoff : float
        Low cutoff frequency in Hz (default: 300)
    high_cutoff : float
        High cutoff frequency in Hz (default: 3400)
    order : int
        Filter order (default: 4)
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Validate cutoffs against Nyquist
    nyquist = sample_rate / 2
    low_cutoff = max(20, min(low_cutoff, nyquist * 0.9))
    high_cutoff = max(low_cutoff + 100, min(high_cutoff, nyquist * 0.95))
    
    # Design bandpass filter using SOS for numerical stability
    sos, _ = _design_butterworth_filter(
        [low_cutoff, high_cutoff],
        sample_rate,
        filter_type='band',
        order=order,
    )
    
    # Apply zero-phase filtering
    output = _apply_filter_zerophase(audio, sos)
    
    return output.astype(np.float32)


def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = 80.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply highpass filter to remove low-frequency noise.
    
    Removes rumble, hum, wind noise, and other low-frequency
    interference while preserving speech fundamentals.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    cutoff : float
        Cutoff frequency in Hz (default: 80)
    order : int
        Filter order (default: 4)
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Validate cutoff
    nyquist = sample_rate / 2
    cutoff = max(10, min(cutoff, nyquist * 0.9))
    
    # Design highpass filter using SOS
    sos, _ = _design_butterworth_filter(
        cutoff,
        sample_rate,
        filter_type='high',
        order=order,
    )
    
    output = _apply_filter_zerophase(audio, sos)
    
    return output.astype(np.float32)


def lowpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply lowpass filter to remove high-frequency noise.
    
    Removes hiss, high-frequency interference, and aliasing
    artifacts while preserving speech harmonics.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    cutoff : float
        Cutoff frequency in Hz (default: 8000)
    order : int
        Filter order (default: 4)
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Validate cutoff
    nyquist = sample_rate / 2
    cutoff = max(100, min(cutoff, nyquist * 0.95))
    
    # Design lowpass filter using SOS
    sos, _ = _design_butterworth_filter(
        cutoff,
        sample_rate,
        filter_type='low',
        order=order,
    )
    
    output = _apply_filter_zerophase(audio, sos)
    
    return output.astype(np.float32)


def notch_filter(
    audio: np.ndarray,
    sample_rate: int,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency (e.g., mains hum).
    
    Removes a narrow band around the notch frequency while
    leaving the rest of the spectrum intact.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    notch_freq : float
        Frequency to notch out in Hz (default: 50 for EU mains)
    quality_factor : float
        Q factor - higher = narrower notch (default: 30)
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Design notch filter
    nyquist = sample_rate / 2
    normalized_freq = notch_freq / nyquist
    
    # Ensure valid frequency
    normalized_freq = max(0.001, min(0.999, normalized_freq))
    
    b, a = signal.iirnotch(normalized_freq, quality_factor)
    
    output = _apply_filter_zerophase(audio, b, a)
    
    return output.astype(np.float32)


def multiband_filter(
    audio: np.ndarray,
    sample_rate: int,
    bands: Optional[list] = None,
    gains_db: Optional[list] = None,
) -> np.ndarray:
    """
    Apply multiband filter with adjustable gains per band.
    
    Splits signal into frequency bands and applies different
    gains to each, useful for targeted noise reduction.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    bands : list
        Band edge frequencies in Hz
        Default: [0, 300, 1000, 4000, nyquist]
    gains_db : list
        Gain for each band in dB
        Default: [-6, 0, 0, -3]
        
    Returns
    -------
    np.ndarray
        Filtered audio signal
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    nyquist = sample_rate / 2
    
    if bands is None:
        bands = [0, 300, 1000, 4000, nyquist]
    if gains_db is None:
        gains_db = [-6, 0, 0, -3]  # Reduce low and high frequencies
    
    # Ensure we have correct number of gains
    assert len(gains_db) == len(bands) - 1, "Need one gain per band"
    
    # Convert gains to linear
    gains = [10 ** (g / 20) for g in gains_db]
    
    output = np.zeros_like(audio)
    
    for i in range(len(gains)):
        low = bands[i]
        high = bands[i + 1]
        gain = gains[i]
        
        if low == 0:
            # Lowpass for first band
            b, a = _design_butterworth_filter(high, sample_rate, 'low', order=4)
        elif high >= nyquist * 0.99:
            # Highpass for last band
            b, a = _design_butterworth_filter(low, sample_rate, 'high', order=4)
        else:
            # Bandpass for middle bands
            b, a = _design_butterworth_filter([low, high], sample_rate, 'band', order=4)
        
        band_signal = _apply_filter_zerophase(audio, b, a)
        output += band_signal * gain
    
    return output.astype(np.float32)
