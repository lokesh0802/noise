"""
RNNoise Noise Suppression.

Wraps the RNNoise library from Xiph.org for real-time noise suppression
using recurrent neural networks.

Theory:
-------
RNNoise uses a hybrid approach combining classical signal processing
with deep learning:

1. **Feature Extraction:** Computes 22 Bark-scale bands plus pitch features
2. **GRU Network:** Three GRU layers process temporal context
3. **Gain Prediction:** Network outputs per-band gains (0-1)
4. **Pitch Filtering:** Comb filter for pitch enhancement

Key advantages:
- Very low latency (~10ms)
- Low computational cost (designed for real-time)
- Trained on diverse noise types
- No voice activity detection needed

The model was trained on 100+ hours of speech with various noise types.

Reference: Valin, J.M. (2018). A Hybrid DSP/Deep Learning Approach to 
Real-Time Full-Band Speech Enhancement. https://arxiv.org/abs/1709.08243

Note: RNNoise requires platform-specific native libraries. If unavailable,
this module provides a scipy-based fallback implementation that approximates
the noise suppression behavior using traditional DSP techniques.

Inspired by: https://github.com/exotel/Agent-Stream/blob/main/engines/audio_enhancer.py
"""

import numpy as np
from typing import Optional, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

# Global state for availability check
_rnnoise_available: Optional[bool] = None
_rnnoise_error: Optional[str] = None
_fallback_available: Optional[bool] = None


def _check_rnnoise_available() -> Tuple[bool, Optional[str]]:
    """
    Check if RNNoise Python bindings are available.
    
    Tries multiple known Python wrappers for RNNoise.
    
    Returns
    -------
    Tuple[bool, str]
        (is_available, error_message)
    """
    global _rnnoise_available, _rnnoise_error
    
    if _rnnoise_available is not None:
        return _rnnoise_available, _rnnoise_error
    
    # Try rnnoise (pip install rnnoise)
    try:
        import rnnoise
        _rnnoise_available = True
        _rnnoise_error = None
        return True, None
    except ImportError:
        pass
    
    # Try pyrnnoise
    try:
        import pyrnnoise
        _rnnoise_available = True
        _rnnoise_error = None
        return True, None
    except ImportError:
        pass
    
    # Try rnnoise_python
    try:
        import rnnoise_python
        _rnnoise_available = True
        _rnnoise_error = None
        return True, None
    except ImportError:
        pass
    
    # Try rnnoiselib
    try:
        import rnnoiselib
        _rnnoise_available = True
        _rnnoise_error = None
        return True, None
    except ImportError:
        pass
    
    _rnnoise_available = False
    _rnnoise_error = (
        "RNNoise native library not available.\n\n"
        "⚠️  Native RNNoise requires manual installation:\n\n"
        "Option 1 - macOS (via Homebrew):\n"
        "   brew install xiph/xiph/rnnoise\n"
        "   Then try: pip install audio-noise-suppressor\n\n"
        "Option 2 - Build from source:\n"
        "   git clone https://github.com/xiph/rnnoise\n"
        "   cd rnnoise\n"
        "   ./autogen.sh && ./configure && make && sudo make install\n\n"
        "Option 3 - Try pre-built Python wrappers:\n"
        "   pip install rnnoise-wrapper\n\n"
        "💡 Fallback mode is available using scipy-based noise suppression."
    )
    return False, _rnnoise_error


def _check_fallback_available() -> bool:
    """Check if the scipy-based fallback is available."""
    global _fallback_available
    
    if _fallback_available is not None:
        return _fallback_available
    
    try:
        from scipy.signal import butter, filtfilt, wiener
        from scipy.fft import fft, ifft
        _fallback_available = True
    except ImportError:
        _fallback_available = False
    
    return _fallback_available


def _get_rnnoise_module():
    """
    Get the available RNNoise module.
    
    Returns the first available RNNoise Python binding.
    """
    try:
        import rnnoise
        return rnnoise, "rnnoise"
    except ImportError:
        pass
    
    try:
        import pyrnnoise
        return pyrnnoise, "pyrnnoise"
    except ImportError:
        pass
    
    try:
        import rnnoise_python
        return rnnoise_python, "rnnoise_python"
    except ImportError:
        pass
    
    return None, None


def _process_with_scipy_fallback(
    audio: np.ndarray,
    sample_rate: int,
    highpass_cutoff: int = 100,
    lowpass_cutoff: int = 8000,
    noise_reduction_factor: float = 2.0,
) -> np.ndarray:
    """
    Process audio using scipy-based fallback implementation.
    
    This approximates RNNoise behavior using traditional DSP techniques:
    1. High-pass filter to remove low-frequency noise (hum, rumble)
    2. Low-pass filter to remove high-frequency noise (hiss)
    3. Spectral subtraction for noise reduction
    4. Wiener filter for final smoothing
    
    Based on approach from: https://github.com/exotel/Agent-Stream
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono, float32/64)
    sample_rate : int
        Sample rate in Hz
    highpass_cutoff : int
        High-pass filter cutoff frequency (default: 100Hz)
    lowpass_cutoff : int
        Low-pass filter cutoff frequency (default: 8000Hz)
    noise_reduction_factor : float
        Strength of spectral subtraction (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Processed audio signal
    """
    from scipy.signal import butter, filtfilt, wiener
    from scipy.fft import fft, ifft
    
    logger.debug("Using scipy-based fallback for RNNoise-style processing")
    
    # Ensure float32
    audio = audio.astype(np.float32)
    original_length = len(audio)
    
    # Step 1: High-pass filter to remove low-frequency noise
    nyquist = sample_rate / 2
    if highpass_cutoff < nyquist:
        hp_normalized = highpass_cutoff / nyquist
        if hp_normalized < 1.0:
            try:
                b_hp, a_hp = butter(4, hp_normalized, btype='high')
                audio = filtfilt(b_hp, a_hp, audio)
            except Exception as e:
                logger.warning(f"High-pass filter failed: {e}")
    
    # Step 2: Low-pass filter to remove high-frequency noise
    if lowpass_cutoff < nyquist:
        lp_normalized = lowpass_cutoff / nyquist
        if lp_normalized < 1.0 and lp_normalized > 0:
            try:
                b_lp, a_lp = butter(4, lp_normalized, btype='low')
                audio = filtfilt(b_lp, a_lp, audio)
            except Exception as e:
                logger.warning(f"Low-pass filter failed: {e}")
    
    # Step 3: Spectral subtraction for noise reduction
    try:
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        if frame_length > 0 and len(audio) > frame_length:
            # Estimate noise from first 0.5 seconds (or less if audio is short)
            noise_frames = min(int(0.5 * sample_rate), len(audio) // 4)
            noise_spectrum = np.abs(fft(audio[:max(noise_frames, frame_length)]))
            
            enhanced_audio = []
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                
                # Apply window
                windowed = frame * np.hanning(len(frame))
                
                # FFT
                spectrum = fft(windowed)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                
                # Spectral subtraction
                enhanced_magnitude = magnitude - noise_reduction_factor * noise_spectrum[:len(magnitude)]
                enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)  # Spectral floor
                
                # Reconstruct
                enhanced_spectrum = enhanced_magnitude * np.exp(1j * phase)
                enhanced_frame = np.real(ifft(enhanced_spectrum))
                
                enhanced_audio.extend(enhanced_frame)
            
            if len(enhanced_audio) > 0:
                audio = np.array(enhanced_audio[:original_length], dtype=np.float32)
                
    except Exception as e:
        logger.warning(f"Spectral subtraction failed: {e}")
    
    # Step 4: Apply Wiener filter for final smoothing
    try:
        # Wiener filter with small noise variance estimate
        wiener_size = min(5, len(audio) // 10) if len(audio) > 50 else None
        if wiener_size and wiener_size > 0:
            audio = wiener(audio, mysize=wiener_size)
    except Exception as e:
        logger.warning(f"Wiener filter failed: {e}")
    
    # Step 5: Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.95
    
    # Ensure output length matches input
    if len(audio) > original_length:
        audio = audio[:original_length]
    elif len(audio) < original_length:
        audio = np.pad(audio, (0, original_length - len(audio)))
    
    return audio.astype(np.float32)


def _process_with_rnnoise_wrapper(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Process audio using rnnoise-wrapper package.
    
    The rnnoise-wrapper package provides a simple interface to RNNoise.
    It expects 16kHz mono audio.
    """
    import rnnoise
    
    # RNNoise expects 16kHz audio
    if sample_rate != 48000:
        from scipy import signal as scipy_signal
        from math import gcd
        
        # Resample to 48kHz (RNNoise native rate for rnnoise-wrapper)
        g = gcd(sample_rate, 48000)
        up = 48000 // g
        down = sample_rate // g
        audio_48k = scipy_signal.resample_poly(audio, up, down)
    else:
        audio_48k = audio
    
    # Process with RNNoise
    # rnnoise-wrapper expects float32 in [-1, 1] range
    audio_48k = audio_48k.astype(np.float32)
    
    # Create denoiser instance
    denoiser = rnnoise.RNNoise()
    
    # Process in frames (RNNoise uses 10ms frames at 48kHz = 480 samples)
    frame_size = 480
    n_frames = len(audio_48k) // frame_size
    
    output = np.zeros_like(audio_48k)
    
    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio_48k[start:end]
        
        # Process frame
        denoised_frame = denoiser.process_frame(frame)
        output[start:end] = denoised_frame
    
    # Handle remaining samples
    remaining = len(audio_48k) % frame_size
    if remaining > 0:
        # Pad last frame
        last_frame = np.zeros(frame_size, dtype=np.float32)
        last_frame[:remaining] = audio_48k[-remaining:]
        denoised_last = denoiser.process_frame(last_frame)
        output[-remaining:] = denoised_last[:remaining]
    
    # Resample back to original rate
    if sample_rate != 48000:
        g = gcd(48000, sample_rate)
        up = sample_rate // g
        down = 48000 // g
        output = scipy_signal.resample_poly(output, up, down)
    
    # Match original length
    if len(output) > len(audio):
        output = output[:len(audio)]
    elif len(output) < len(audio):
        output = np.pad(output, (0, len(audio) - len(output)))
    
    return output


def _process_with_pyrnnoise(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Process audio using pyrnnoise package.
    """
    import pyrnnoise
    
    # pyrnnoise typically expects 48kHz
    target_sr = 48000
    
    if sample_rate != target_sr:
        from scipy import signal as scipy_signal
        from math import gcd
        
        g = gcd(sample_rate, target_sr)
        up = target_sr // g
        down = sample_rate // g
        audio_resampled = scipy_signal.resample_poly(audio, up, down)
    else:
        audio_resampled = audio
    
    # Convert to int16 if needed by the library
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    
    # Process
    denoiser = pyrnnoise.RNNoise()
    output_int16 = denoiser.filter(audio_int16)
    
    # Convert back to float
    output = output_int16.astype(np.float32) / 32767.0
    
    # Resample back
    if sample_rate != target_sr:
        g = gcd(target_sr, sample_rate)
        up = sample_rate // g
        down = target_sr // g
        output = scipy_signal.resample_poly(output, up, down)
    
    # Match original length
    if len(output) > len(audio):
        output = output[:len(audio)]
    elif len(output) < len(audio):
        output = np.pad(output, (0, len(audio) - len(output)))
    
    return output


def rnnoise_suppress(
    audio: np.ndarray,
    sample_rate: int,
    use_fallback: bool = False,
) -> np.ndarray:
    """
    Apply RNNoise noise suppression.
    
    RNNoise is a recurrent neural network-based noise suppressor from
    Xiph.org, designed for real-time low-latency processing.
    
    If the native RNNoise library is not available and use_fallback=True,
    a scipy-based approximation will be used instead.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono, float32/64)
    sample_rate : int
        Sample rate in Hz
    use_fallback : bool
        If True, use scipy-based fallback when native RNNoise unavailable
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
        
    Raises
    ------
    RuntimeError
        If RNNoise is not available and fallback is disabled
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Check native RNNoise availability
    available, error = _check_rnnoise_available()
    
    if available:
        # Use native RNNoise
        module, module_name = _get_rnnoise_module()
        
        if module is None:
            if use_fallback and _check_fallback_available():
                logger.info("Native RNNoise module failed to load, using scipy fallback")
                return _process_with_scipy_fallback(audio, sample_rate)
            raise RuntimeError("RNNoise module could not be loaded")
        
        # Process based on which module is available
        try:
            if module_name == "rnnoise":
                output = _process_with_rnnoise_wrapper(audio, sample_rate)
            elif module_name == "pyrnnoise":
                output = _process_with_pyrnnoise(audio, sample_rate)
            else:
                # Generic fallback - try common API patterns
                if hasattr(module, 'RNNoise'):
                    denoiser = module.RNNoise()
                    if hasattr(denoiser, 'process_frame'):
                        output = _process_with_rnnoise_wrapper(audio, sample_rate)
                    elif hasattr(denoiser, 'filter'):
                        output = _process_with_pyrnnoise(audio, sample_rate)
                    else:
                        raise RuntimeError(f"Unknown RNNoise API in {module_name}")
                else:
                    raise RuntimeError(f"Unknown RNNoise module structure: {module_name}")
            
            return output.astype(np.float32)
            
        except Exception as e:
            if use_fallback and _check_fallback_available():
                logger.warning(f"Native RNNoise failed: {e}, using scipy fallback")
                return _process_with_scipy_fallback(audio, sample_rate)
            raise RuntimeError(f"RNNoise processing failed: {str(e)}")
    
    else:
        # Native RNNoise not available
        if use_fallback and _check_fallback_available():
            logger.info("Native RNNoise not available, using scipy fallback")
            return _process_with_scipy_fallback(audio, sample_rate)
        
        raise RuntimeError(
            f"RNNoise is not supported on this platform.\n\n{error}"
        )


def rnnoise_suppress_fallback(
    audio: np.ndarray,
    sample_rate: int,
    highpass_cutoff: int = 100,
    lowpass_cutoff: int = 8000,
    noise_reduction_factor: float = 2.0,
) -> np.ndarray:
    """
    Apply RNNoise-style noise suppression using scipy-based fallback.
    
    This provides a noise suppression method similar to RNNoise using
    traditional DSP techniques when the native library is unavailable.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono, float32/64)
    sample_rate : int
        Sample rate in Hz
    highpass_cutoff : int
        High-pass filter cutoff frequency (default: 100Hz)
    lowpass_cutoff : int  
        Low-pass filter cutoff frequency (default: 8000Hz)
    noise_reduction_factor : float
        Strength of spectral subtraction (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Denoised audio signal
        
    Raises
    ------
    RuntimeError
        If scipy is not available
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    if not _check_fallback_available():
        raise RuntimeError("scipy is required for fallback mode. Install with: pip install scipy")
    
    return _process_with_scipy_fallback(
        audio, sample_rate, 
        highpass_cutoff, lowpass_cutoff, 
        noise_reduction_factor
    )


def is_rnnoise_available(include_fallback: bool = False) -> bool:
    """
    Check if RNNoise is available on this platform.
    
    Parameters
    ----------
    include_fallback : bool
        If True, also return True when scipy fallback is available
    
    Returns
    -------
    bool
        True if RNNoise (or fallback) can be used
    """
    available, _ = _check_rnnoise_available()
    if available:
        return True
    
    if include_fallback:
        return _check_fallback_available()
    
    return False


def is_fallback_available() -> bool:
    """
    Check if the scipy-based fallback is available.
    
    Returns
    -------
    bool
        True if fallback mode can be used
    """
    return _check_fallback_available()


def get_rnnoise_status() -> dict:
    """
    Get detailed status of RNNoise availability.
    
    Returns
    -------
    dict
        Status information including availability and error message
    """
    available, error = _check_rnnoise_available()
    module, module_name = _get_rnnoise_module() if available else (None, None)
    fallback_available = _check_fallback_available()
    
    return {
        "available": available,
        "module_name": module_name,
        "fallback_available": fallback_available,
        "any_available": available or fallback_available,
        "error_message": error,
        "install_hint": (
            "Build from source: https://github.com/xiph/rnnoise\n"
            "Or try: pip install rnnoise (if available)\n"
            "💡 Fallback mode is available!" if fallback_available else
            "Build from source: https://github.com/xiph/rnnoise\n"
            "Or try: pip install rnnoise (if available)"
        ) if not available else None,
    }
