"""
Audio Processing Pipelines.

Allows combining multiple filters and suppressors in sequence
to test different processing strategies.

Example pipelines:
- Highpass → Spectral Gating (remove rumble, then denoise)
- Bandpass → Wiener Filter (isolate voice band, then enhance)
- Spectral Gate → Adaptive Wiener (two-stage denoising)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PipelineStage:
    """A single stage in a processing pipeline."""
    name: str
    processor_key: str
    params: Dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class Pipeline:
    """A complete processing pipeline."""
    name: str
    description: str
    stages: List[PipelineStage]
    category: str = "Custom"


def get_predefined_pipelines() -> Dict[str, Pipeline]:
    """
    Get dictionary of predefined processing pipelines.
    
    Returns
    -------
    Dict[str, Pipeline]
        Dictionary mapping pipeline keys to Pipeline objects
    """
    pipelines = {}
    
    # Voice enhancement pipelines
    pipelines["voice_enhance_basic"] = Pipeline(
        name="Voice Enhancement (Basic)",
        description="Bandpass filter + Spectral gating for clean speech",
        stages=[
            PipelineStage("Bandpass Filter", "bandpass", {}),
            PipelineStage("Spectral Gating", "spectral_gate", {}),
        ],
        category="Voice"
    )
    
    pipelines["voice_enhance_advanced"] = Pipeline(
        name="Voice Enhancement (Advanced)",
        description="Highpass + Wiener + Spectral gating for high-quality speech",
        stages=[
            PipelineStage("Highpass Filter", "highpass", {}),
            PipelineStage("Wiener Filter", "wiener_adaptive", {}),
            PipelineStage("Spectral Gating", "spectral_gate_adaptive", {}),
        ],
        category="Voice"
    )
    
    # Rumble removal pipelines
    pipelines["rumble_removal"] = Pipeline(
        name="Rumble Removal",
        description="Highpass filter + denoising for low-frequency noise",
        stages=[
            PipelineStage("Highpass Filter", "highpass", {}),
            PipelineStage("Spectral Gating", "spectral_gate", {}),
        ],
        category="Noise Reduction"
    )
    
    # Two-stage denoising
    pipelines["two_stage_denoise"] = Pipeline(
        name="Two-Stage Denoising",
        description="Spectral gate followed by Wiener for aggressive noise removal",
        stages=[
            PipelineStage("Spectral Gating", "spectral_gate", {}),
            PipelineStage("Wiener Filter", "wiener_adaptive", {}),
        ],
        category="Noise Reduction"
    )
    
    # Telephone quality simulation
    pipelines["telephone_quality"] = Pipeline(
        name="Telephone Quality",
        description="Bandpass filter to simulate telephone bandwidth",
        stages=[
            PipelineStage("Bandpass Filter", "bandpass", {}),
        ],
        category="Effects"
    )
    
    # Maximum denoising
    pipelines["max_denoise"] = Pipeline(
        name="Maximum Denoising",
        description="Aggressive multi-stage denoising (may affect quality)",
        stages=[
            PipelineStage("Highpass Filter", "highpass", {}),
            PipelineStage("Spectral Gating", "spectral_gate_adaptive", {}),
            PipelineStage("Wiener Filter", "wiener_adaptive", {}),
        ],
        category="Noise Reduction"
    )
    
    # Conservative denoising
    pipelines["conservative_denoise"] = Pipeline(
        name="Conservative Denoising",
        description="Light denoising that preserves audio quality",
        stages=[
            PipelineStage("Spectral Gating", "spectral_gate", {}),
        ],
        category="Noise Reduction"
    )
    
    return pipelines


def run_pipeline(
    pipeline: Pipeline,
    audio: np.ndarray,
    sample_rate: int,
    processor_funcs: Dict[str, Callable],
) -> Tuple[np.ndarray, List[str]]:
    """
    Execute a processing pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute
    audio : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    processor_funcs : Dict[str, Callable]
        Mapping of processor keys to callable functions
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Processed audio and list of stage descriptions
    """
    current_audio = audio.copy()
    stage_descriptions = []
    
    for stage in pipeline.stages:
        if stage.processor_key not in processor_funcs:
            raise ValueError(f"Unknown processor: {stage.processor_key}")
        
        processor = processor_funcs[stage.processor_key]
        
        # Apply processing
        try:
            current_audio = processor(current_audio, sample_rate, **stage.params)
            stage_descriptions.append(f"✓ {stage.name}")
        except Exception as e:
            stage_descriptions.append(f"✗ {stage.name}: {str(e)}")
            raise
    
    return current_audio, stage_descriptions


def create_custom_pipeline(
    name: str,
    description: str,
    processor_keys: List[str],
) -> Pipeline:
    """
    Create a custom pipeline from a list of processor keys.
    
    Parameters
    ----------
    name : str
        Pipeline name
    description : str
        Pipeline description
    processor_keys : List[str]
        List of processor keys in order
        
    Returns
    -------
    Pipeline
        Custom pipeline object
    """
    stages = []
    for key in processor_keys:
        stages.append(PipelineStage(
            name=key.replace("_", " ").title(),
            processor_key=key,
            params={}
        ))
    
    return Pipeline(
        name=name,
        description=description,
        stages=stages,
        category="Custom"
    )


def get_pipeline_description(pipeline: Pipeline) -> str:
    """
    Get a formatted description of a pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to describe
        
    Returns
    -------
    str
        Formatted description
    """
    stages_str = " → ".join([stage.name for stage in pipeline.stages])
    return f"{pipeline.name}\n{stages_str}\n{pipeline.description}"
