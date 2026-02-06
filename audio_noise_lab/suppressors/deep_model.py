"""
Deep Learning-based Noise Suppression.

Implements neural network-based denoising using pretrained models.
Supports SpeechBrain's SepFormer model with graceful fallbacks.

Theory:
-------
Deep learning approaches to speech enhancement have shown significant
improvements over classical methods, especially for non-stationary noise.

SpeechBrain's SepFormer (Separation Transformer):
- Uses self-attention mechanisms for temporal modeling
- Trained on large speech corpora with simulated noise
- Can generalize to unseen noise types
- Operates on raw waveforms (no explicit STFT)

When the model is unavailable, the module provides clear feedback
and can fall back to classical methods.

Note: Deep learning models require significant memory and may be
slow on CPU. GPU acceleration is recommended for real-time use.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class DeepModelStatus:
    """Status information for deep learning models."""
    
    speechbrain_available: bool
    torch_available: bool
    model_loaded: bool
    model_name: str
    error_message: Optional[str]
    fallback_reason: Optional[str]


# Global model cache (loaded lazily)
_model_cache: Dict[str, Any] = {}
_status_cache: Optional[DeepModelStatus] = None


def _check_torch_available() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch is available."""
    try:
        import torch
        return True, None
    except ImportError:
        return False, "PyTorch not installed. Install with: pip install torch"


def _check_speechbrain_available() -> Tuple[bool, Optional[str]]:
    """Check if SpeechBrain is available."""
    try:
        import speechbrain
        return True, None
    except ImportError:
        return False, "SpeechBrain not installed. Install with: pip install speechbrain"
    except Exception as e:
        return False, f"SpeechBrain import failed: {str(e)}. Try: pip install --upgrade torchaudio speechbrain"


def _load_speechbrain_model() -> Tuple[Any, Optional[str]]:
    """
    Load SpeechBrain's pretrained speech enhancement model.
    
    Returns
    -------
    Tuple[model, error_message]
        The loaded model or None, and any error message
    """
    global _model_cache
    
    if "speechbrain_enhance" in _model_cache:
        return _model_cache["speechbrain_enhance"], None
    
    try:
        from speechbrain.inference.separation import SepformerSeparation
        
        # Load pretrained model (downloads automatically if needed)
        # Using the speech enhancement model trained on WHAMR!
        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr-enhancement",
            savedir="pretrained_models/sepformer-whamr-enhancement",
        )
        
        _model_cache["speechbrain_enhance"] = model
        return model, None
        
    except Exception as e:
        error_msg = f"Failed to load SpeechBrain model: {str(e)}"
        return None, error_msg


def _load_demucs_model() -> Tuple[Any, Optional[str]]:
    """
    Load Facebook's Demucs model for speech enhancement.
    
    Demucs is primarily a music source separation model but can
    be used for speech enhancement when isolating voice.
    
    Returns
    -------
    Tuple[model, error_message]
    """
    global _model_cache
    
    if "demucs" in _model_cache:
        return _model_cache["demucs"], None
    
    try:
        # Try to import demucs
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch
        
        # Load hybrid demucs model
        model = get_model("htdemucs")
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        _model_cache["demucs"] = model
        _model_cache["demucs_apply"] = apply_model
        return model, None
        
    except ImportError:
        return None, "Demucs not installed. Install with: pip install demucs"
    except Exception as e:
        return None, f"Failed to load Demucs model: {str(e)}"


def get_deep_model_status() -> DeepModelStatus:
    """
    Get the status of deep learning models and dependencies.
    
    Returns detailed information about what's available and
    why certain models may not be usable.
    
    Returns
    -------
    DeepModelStatus
        Status information for display in UI
    """
    global _status_cache
    
    # Check basic dependencies
    torch_available, torch_error = _check_torch_available()
    sb_available, sb_error = _check_speechbrain_available()
    
    fallback_reason = None
    model_loaded = False
    model_name = "None"
    error_message = None
    
    if not torch_available:
        fallback_reason = torch_error
        error_message = torch_error
    elif not sb_available:
        # Try demucs as fallback
        demucs_model, demucs_error = _load_demucs_model()
        if demucs_model is not None:
            model_loaded = True
            model_name = "Demucs (htdemucs)"
        else:
            fallback_reason = f"SpeechBrain: {sb_error}\nDemucs: {demucs_error}"
            error_message = sb_error
    else:
        # Try to load SpeechBrain model
        sb_model, sb_load_error = _load_speechbrain_model()
        if sb_model is not None:
            model_loaded = True
            model_name = "SpeechBrain SepFormer (WHAMR)"
        else:
            # Try demucs as fallback
            demucs_model, demucs_error = _load_demucs_model()
            if demucs_model is not None:
                model_loaded = True
                model_name = "Demucs (htdemucs)"
            else:
                fallback_reason = f"SpeechBrain load error: {sb_load_error}"
                error_message = sb_load_error
    
    _status_cache = DeepModelStatus(
        speechbrain_available=sb_available,
        torch_available=torch_available,
        model_loaded=model_loaded,
        model_name=model_name,
        error_message=error_message,
        fallback_reason=fallback_reason,
    )
    
    return _status_cache


def _enhance_with_speechbrain(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Enhance audio using SpeechBrain's SepFormer model.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio (mono, any sample rate)
    sample_rate : int
        Input sample rate
        
    Returns
    -------
    np.ndarray
        Enhanced audio
    """
    import torch
    from scipy import signal as scipy_signal
    
    model, error = _load_speechbrain_model()
    if model is None:
        raise RuntimeError(f"SpeechBrain model not available: {error}")
    
    # Model expects 8kHz audio
    target_sr = 8000
    
    # Resample if needed
    if sample_rate != target_sr:
        from math import gcd
        g = gcd(sample_rate, target_sr)
        up = target_sr // g
        down = sample_rate // g
        audio_resampled = scipy_signal.resample_poly(audio, up, down)
    else:
        audio_resampled = audio
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio_resampled, dtype=torch.float32).unsqueeze(0)
    
    # Run enhancement
    with torch.no_grad():
        enhanced = model.separate_batch(audio_tensor)
    
    # Get enhanced signal (first source)
    enhanced_np = enhanced[0, :, 0].cpu().numpy()
    
    # Resample back to original rate
    if sample_rate != target_sr:
        g = gcd(target_sr, sample_rate)
        up = sample_rate // g
        down = target_sr // g
        enhanced_np = scipy_signal.resample_poly(enhanced_np, up, down)
    
    # Match original length
    if len(enhanced_np) > len(audio):
        enhanced_np = enhanced_np[:len(audio)]
    elif len(enhanced_np) < len(audio):
        enhanced_np = np.pad(enhanced_np, (0, len(audio) - len(enhanced_np)))
    
    return enhanced_np


def _enhance_with_demucs(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Enhance audio using Demucs source separation.
    
    Extracts the 'vocals' source which contains the clean speech.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio (mono)
    sample_rate : int
        Sample rate
        
    Returns
    -------
    np.ndarray
        Enhanced audio (vocals track)
    """
    import torch
    from scipy import signal as scipy_signal
    
    model, error = _load_demucs_model()
    if model is None:
        raise RuntimeError(f"Demucs model not available: {error}")
    
    apply_model = _model_cache.get("demucs_apply")
    
    # Demucs expects 44.1kHz stereo
    target_sr = 44100
    
    # Resample if needed
    if sample_rate != target_sr:
        from math import gcd
        g = gcd(sample_rate, target_sr)
        up = target_sr // g
        down = sample_rate // g
        audio_resampled = scipy_signal.resample_poly(audio, up, down)
    else:
        audio_resampled = audio
    
    # Convert to stereo tensor (batch, channels, samples)
    audio_stereo = np.stack([audio_resampled, audio_resampled])
    audio_tensor = torch.tensor(audio_stereo, dtype=torch.float32).unsqueeze(0)
    
    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()
    
    # Run separation
    with torch.no_grad():
        sources = apply_model(model, audio_tensor)
    
    # Get vocals (index 3 in standard demucs)
    # Sources order: drums, bass, other, vocals
    vocals = sources[0, 3, 0, :].cpu().numpy()  # Take left channel
    
    # Resample back
    if sample_rate != target_sr:
        g = gcd(target_sr, sample_rate)
        up = sample_rate // g
        down = target_sr // g
        vocals = scipy_signal.resample_poly(vocals, up, down)
    
    # Match original length
    if len(vocals) > len(audio):
        vocals = vocals[:len(audio)]
    elif len(vocals) < len(audio):
        vocals = np.pad(vocals, (0, len(audio) - len(vocals)))
    
    return vocals


def deep_denoise(
    audio: np.ndarray,
    sample_rate: int,
    model_preference: str = "auto",
) -> np.ndarray:
    """
    Apply deep learning-based noise suppression.
    
    Automatically selects the best available model:
    1. SpeechBrain SepFormer (preferred for speech)
    2. Demucs (fallback, better for music)
    
    If no deep learning model is available, raises an error
    with clear instructions for installation.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (mono)
    sample_rate : int
        Sample rate in Hz
    model_preference : str
        Model to use: 'auto', 'speechbrain', or 'demucs'
        
    Returns
    -------
    np.ndarray
        Enhanced audio signal
        
    Raises
    ------
    RuntimeError
        If no suitable model is available
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Get status to determine what's available
    status = get_deep_model_status()
    
    if not status.torch_available:
        raise RuntimeError(
            "Deep learning models require PyTorch.\n"
            "Install with: pip install torch\n"
            "For GPU support, see: https://pytorch.org/get-started/locally/"
        )
    
    if not status.model_loaded:
        raise RuntimeError(
            f"No deep learning model available.\n"
            f"Reason: {status.fallback_reason}\n\n"
            "To enable deep learning denoising, install one of:\n"
            "1. SpeechBrain: pip install speechbrain\n"
            "2. Demucs: pip install demucs\n\n"
            "SpeechBrain is recommended for speech enhancement."
        )
    
    # Select model based on preference and availability
    if model_preference == "speechbrain" and status.speechbrain_available:
        try:
            return _enhance_with_speechbrain(audio, sample_rate).astype(np.float32)
        except Exception as e:
            warnings.warn(f"SpeechBrain failed: {e}, trying fallback")
    
    if model_preference == "demucs" and "demucs" in _model_cache:
        try:
            return _enhance_with_demucs(audio, sample_rate).astype(np.float32)
        except Exception as e:
            warnings.warn(f"Demucs failed: {e}, trying fallback")
    
    # Auto mode - try available models in order
    if "speechbrain_enhance" in _model_cache:
        try:
            return _enhance_with_speechbrain(audio, sample_rate).astype(np.float32)
        except Exception as e:
            warnings.warn(f"SpeechBrain failed: {e}, trying Demucs")
    
    if "demucs" in _model_cache:
        try:
            return _enhance_with_demucs(audio, sample_rate).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"All deep learning models failed. Last error: {e}")
    
    raise RuntimeError("No deep learning model could be loaded.")


def is_deep_learning_available() -> bool:
    """
    Check if any deep learning model is available.
    
    Returns
    -------
    bool
        True if at least one model can be used
    """
    status = get_deep_model_status()
    return status.model_loaded


def get_available_models() -> list:
    """
    Get list of available deep learning models.
    
    Returns
    -------
    list
        Names of available models
    """
    models = []
    
    torch_ok, _ = _check_torch_available()
    if not torch_ok:
        return models
    
    sb_ok, _ = _check_speechbrain_available()
    if sb_ok:
        models.append("speechbrain")
    
    try:
        import demucs
        models.append("demucs")
    except ImportError:
        pass
    
    return models
