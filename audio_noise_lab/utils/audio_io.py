"""
Audio I/O utilities for loading, saving, and preprocessing audio files.

Supports WAV, MP3, and FLAC formats with automatic format detection.
Handles mono/stereo conversion and resampling safely.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa


@dataclass
class AudioData:
    """Container for audio data with metadata."""
    
    samples: np.ndarray  # Shape: (n_samples,) for mono, (n_samples, n_channels) for stereo
    sample_rate: int
    original_sample_rate: int
    is_mono: bool
    duration_seconds: float
    bit_depth: Optional[int] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the audio."""
        return self.samples.shape[0]
    
    @property
    def n_channels(self) -> int:
        """Number of audio channels."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]


def load_audio(
    source: Union[str, Path, BytesIO, bytes],
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    dtype: np.dtype = np.float32,
) -> AudioData:
    """
    Load audio from file or bytes with optional resampling and normalization.
    
    Parameters
    ----------
    source : str, Path, BytesIO, or bytes
        Audio file path or binary data
    target_sr : int
        Target sample rate for resampling (default: 16000 Hz)
    mono : bool
        Convert to mono if True (default: True)
    normalize : bool
        Normalize audio to [-1, 1] range (default: True)
    dtype : np.dtype
        Output data type (default: np.float32)
        
    Returns
    -------
    AudioData
        Loaded and preprocessed audio data
        
    Raises
    ------
    ValueError
        If audio file is empty or corrupted
    IOError
        If file format is not supported
    """
    # Handle bytes input
    if isinstance(source, bytes):
        source = BytesIO(source)
    
    # Load audio using soundfile (handles WAV, FLAC, OGG natively)
    try:
        samples, original_sr = sf.read(source, dtype='float64')
    except Exception as e:
        # Try with pydub for MP3 support
        try:
            from pydub import AudioSegment
            
            if isinstance(source, BytesIO):
                source.seek(0)
                audio_segment = AudioSegment.from_file(source)
            else:
                audio_segment = AudioSegment.from_file(source)
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
            original_sr = audio_segment.frame_rate
            
            # Handle stereo from pydub (interleaved)
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Normalize to [-1, 1] range (pydub uses int16)
            if audio_segment.sample_width == 2:
                samples = samples / 32768.0
            elif audio_segment.sample_width == 1:
                samples = (samples - 128) / 128.0
            elif audio_segment.sample_width == 4:
                samples = samples / 2147483648.0
                
        except ImportError:
            raise IOError(
                f"Could not load audio: {e}. "
                "For MP3 support, install pydub and ffmpeg."
            )
        except Exception as e2:
            raise IOError(f"Failed to load audio file: {e}; {e2}")
    
    # Validate audio
    if samples.size == 0:
        raise ValueError("Audio file is empty")
    
    if not np.isfinite(samples).all():
        # Replace NaN/Inf with zeros
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Convert to mono if requested
    if mono and samples.ndim > 1:
        samples = to_mono(samples)
    
    is_mono = samples.ndim == 1
    
    # Resample if needed
    if original_sr != target_sr:
        samples = resample_audio(samples, original_sr, target_sr)
    
    # Normalize if requested
    if normalize:
        samples = normalize_audio(samples)
    
    # Convert to target dtype
    samples = samples.astype(dtype)
    
    duration = len(samples) / target_sr if is_mono else len(samples) / target_sr
    
    return AudioData(
        samples=samples,
        sample_rate=target_sr,
        original_sample_rate=original_sr,
        is_mono=is_mono,
        duration_seconds=duration,
    )


def save_audio(
    samples: np.ndarray,
    sample_rate: int,
    path: Optional[Union[str, Path]] = None,
    format: str = "WAV",
    subtype: str = "PCM_16",
) -> Optional[bytes]:
    """
    Save audio to file or return as bytes.
    
    Parameters
    ----------
    samples : np.ndarray
        Audio samples (mono or stereo)
    sample_rate : int
        Sample rate in Hz
    path : str, Path, or None
        Output file path. If None, returns bytes.
    format : str
        Output format (WAV, FLAC, OGG)
    subtype : str
        Audio subtype (PCM_16, PCM_24, FLOAT)
        
    Returns
    -------
    bytes or None
        If path is None, returns audio as bytes
    """
    # Ensure samples are in valid range
    samples = np.clip(samples, -1.0, 1.0)
    
    if path is None:
        # Return as bytes
        buffer = BytesIO()
        sf.write(buffer, samples, sample_rate, format=format, subtype=subtype)
        buffer.seek(0)
        return buffer.read()
    else:
        sf.write(path, samples, sample_rate, subtype=subtype)
        return None


def normalize_audio(
    samples: np.ndarray,
    target_peak: float = 0.95,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Normalize audio to target peak amplitude.
    
    Uses peak normalization to preserve dynamic range while
    preventing clipping.
    
    Parameters
    ----------
    samples : np.ndarray
        Input audio samples
    target_peak : float
        Target peak amplitude (default: 0.95 to leave headroom)
    eps : float
        Small value to prevent division by zero
        
    Returns
    -------
    np.ndarray
        Normalized audio samples
    """
    peak = np.max(np.abs(samples))
    if peak < eps:
        return samples
    
    return samples * (target_peak / peak)


def resample_audio(
    samples: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate using high-quality resampling.
    
    Uses librosa.resample which provides kaiser_best quality by default,
    offering excellent anti-aliasing with minimal artifacts.
    
    Parameters
    ----------
    samples : np.ndarray
        Input audio samples
    original_sr : int
        Original sample rate in Hz
    target_sr : int
        Target sample rate in Hz
        
    Returns
    -------
    np.ndarray
        Resampled audio samples
    """
    if original_sr == target_sr:
        return samples
    
    # Apply resampling to each channel if stereo
    if samples.ndim == 1:
        return librosa.resample(
            samples, 
            orig_sr=original_sr, 
            target_sr=target_sr,
            res_type='kaiser_best'
        )
    else:
        resampled_channels = []
        for ch in range(samples.shape[1]):
            resampled_ch = librosa.resample(
                samples[:, ch],
                orig_sr=original_sr,
                target_sr=target_sr,
                res_type='kaiser_best'
            )
            resampled_channels.append(resampled_ch)
        return np.column_stack(resampled_channels)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Parameters
    ----------
    samples : np.ndarray
        Input audio samples, shape (n_samples, n_channels)
        
    Returns
    -------
    np.ndarray
        Mono audio samples, shape (n_samples,)
    """
    if samples.ndim == 1:
        return samples
    
    return np.mean(samples, axis=1)


def get_audio_info(source: Union[str, Path, BytesIO]) -> dict:
    """
    Get audio file information without loading full data.
    
    Parameters
    ----------
    source : str, Path, or BytesIO
        Audio file path or binary data
        
    Returns
    -------
    dict
        Audio metadata including duration, sample_rate, channels
    """
    try:
        info = sf.info(source)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
        }
    except Exception:
        return {}
