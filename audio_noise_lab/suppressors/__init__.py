"""
Noise suppression algorithms module.

Each suppressor is implemented as a pure function with signature:
    input: np.ndarray, sample_rate: int -> np.ndarray

No global state is modified.
"""

from .spectral_gate import spectral_gate, spectral_gate_adaptive
from .wiener import wiener_filter, wiener_filter_adaptive
from .bandpass import bandpass_filter, highpass_filter, lowpass_filter
from .noisereduce_wrapper import (
    noisereduce_stationary,
    noisereduce_nonstationary,
)
from .deep_model import deep_denoise, get_deep_model_status

# Registry of all available suppressors
SUPPRESSORS = {
    "spectral_gate": {
        "name": "Spectral Gating",
        "func": spectral_gate,
        "description": "STFT-based spectral gating with noise floor estimation",
        "category": "classical",
    },
    "spectral_gate_adaptive": {
        "name": "Adaptive Spectral Gating",
        "func": spectral_gate_adaptive,
        "description": "Spectral gating with adaptive threshold",
        "category": "classical",
    },
    "wiener": {
        "name": "Wiener Filter",
        "func": wiener_filter,
        "description": "Frequency-domain Wiener filtering",
        "category": "classical",
    },
    "wiener_adaptive": {
        "name": "Adaptive Wiener Filter",
        "func": wiener_filter_adaptive,
        "description": "Wiener filter with adaptive noise estimation",
        "category": "classical",
    },
    "bandpass": {
        "name": "Bandpass Filter (300-3400 Hz)",
        "func": bandpass_filter,
        "description": "Voice-band bandpass filter for speech",
        "category": "filter",
    },
    "highpass": {
        "name": "Highpass Filter (80 Hz)",
        "func": highpass_filter,
        "description": "Removes low-frequency rumble",
        "category": "filter",
    },
    "noisereduce_stationary": {
        "name": "Noisereduce (Stationary)",
        "func": noisereduce_stationary,
        "description": "noisereduce library - stationary noise mode",
        "category": "library",
    },
    "noisereduce_nonstationary": {
        "name": "Noisereduce (Non-stationary)",
        "func": noisereduce_nonstationary,
        "description": "noisereduce library - non-stationary noise mode",
        "category": "library",
    },
    "deep_learning": {
        "name": "Deep Learning (SpeechBrain/Demucs)",
        "func": deep_denoise,
        "description": "Neural network-based denoising",
        "category": "deep_learning",
    },
}

__all__ = [
    "spectral_gate",
    "spectral_gate_adaptive",
    "wiener_filter",
    "wiener_filter_adaptive",
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
    "noisereduce_stationary",
    "noisereduce_nonstationary",
    "deep_denoise",
    "get_deep_model_status",
    "SUPPRESSORS",
]
