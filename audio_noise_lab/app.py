"""
Audio Noise Lab - Professional Audio Noise Suppression Benchmarking.

A Streamlit application for comparing noise suppression algorithms
with research-grade metrics and visualization.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings

# Local imports
from utils.audio_io import load_audio, save_audio, AudioData
from utils.timing import Timer
from analysis.metrics import compute_all_metrics, MetricsResult, interpret_metrics
from analysis.plots import (
    plot_waveform_comparison,
    plot_spectrogram,
    plot_difference_spectrogram,
    plot_full_analysis,
    create_comparison_figure,
    plot_metrics_comparison,
    figure_to_bytes,
)

# Suppress warnings in UI
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Audio Noise Lab",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class SuppressorResult:
    """Container for suppressor processing results."""
    name: str
    processed_audio: np.ndarray
    metrics: MetricsResult
    success: bool
    error_message: Optional[str] = None


def get_available_suppressors() -> Dict[str, dict]:
    """
    Get dictionary of available suppressors with metadata.
    
    Checks for optional dependencies and marks unavailable suppressors.
    """
    suppressors = {}
    
    # Always available (no extra dependencies)
    suppressors["spectral_gate"] = {
        "name": "Spectral Gating",
        "description": "STFT-based spectral gating with noise floor estimation",
        "available": True,
        "category": "Classical",
    }
    
    suppressors["spectral_gate_adaptive"] = {
        "name": "Adaptive Spectral Gating", 
        "description": "Spectral gating with time-varying noise estimation",
        "available": True,
        "category": "Classical",
    }
    
    suppressors["wiener"] = {
        "name": "Wiener Filter",
        "description": "Frequency-domain Wiener filtering with minimum statistics",
        "available": True,
        "category": "Classical",
    }
    
    suppressors["wiener_adaptive"] = {
        "name": "Adaptive Wiener Filter",
        "description": "Decision-directed Wiener filter",
        "available": True,
        "category": "Classical",
    }
    
    suppressors["bandpass"] = {
        "name": "Bandpass Filter (300-3400 Hz)",
        "description": "Voice-band bandpass filter for speech",
        "available": True,
        "category": "Filter",
    }
    
    suppressors["highpass"] = {
        "name": "Highpass Filter (80 Hz)",
        "description": "Removes low-frequency rumble",
        "available": True,
        "category": "Filter",
    }
    
    # Check noisereduce availability
    try:
        import noisereduce
        nr_available = True
    except ImportError:
        nr_available = False
    
    suppressors["noisereduce_stationary"] = {
        "name": "Noisereduce (Stationary)",
        "description": "noisereduce library - for constant background noise",
        "available": nr_available,
        "category": "Library",
        "install_hint": "pip install noisereduce" if not nr_available else None,
    }
    
    suppressors["noisereduce_nonstationary"] = {
        "name": "Noisereduce (Non-stationary)",
        "description": "noisereduce library - for varying noise",
        "available": nr_available,
        "category": "Library",
        "install_hint": "pip install noisereduce" if not nr_available else None,
    }
    
    # Check deep learning availability
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    try:
        import speechbrain
        sb_available = True
    except ImportError:
        sb_available = False
    
    dl_available = torch_available and sb_available
    
    suppressors["deep_learning"] = {
        "name": "Deep Learning (SpeechBrain)",
        "description": "Neural network-based speech enhancement",
        "available": dl_available,
        "category": "Deep Learning",
        "install_hint": "pip install torch speechbrain" if not dl_available else None,
    }
    
    return suppressors


def run_suppressor(
    suppressor_key: str,
    audio: np.ndarray,
    sample_rate: int,
) -> SuppressorResult:
    """
    Run a single suppressor and return results.
    
    Each suppressor is run in isolation to prevent state leakage.
    """
    timer = Timer()
    
    try:
        # Import suppressors here to avoid circular imports
        from suppressors.spectral_gate import spectral_gate, spectral_gate_adaptive
        from suppressors.wiener import wiener_filter, wiener_filter_adaptive
        from suppressors.bandpass import bandpass_filter, highpass_filter
        
        # Map keys to functions
        suppressor_funcs = {
            "spectral_gate": spectral_gate,
            "spectral_gate_adaptive": spectral_gate_adaptive,
            "wiener": wiener_filter,
            "wiener_adaptive": wiener_filter_adaptive,
            "bandpass": bandpass_filter,
            "highpass": highpass_filter,
        }
        
        # Handle noisereduce separately
        if suppressor_key.startswith("noisereduce"):
            from suppressors.noisereduce_wrapper import (
                noisereduce_stationary,
                noisereduce_nonstationary,
            )
            if suppressor_key == "noisereduce_stationary":
                func = noisereduce_stationary
            else:
                func = noisereduce_nonstationary
        elif suppressor_key == "deep_learning":
            from suppressors.deep_model import deep_denoise
            func = deep_denoise
        else:
            func = suppressor_funcs.get(suppressor_key)
        
        if func is None:
            return SuppressorResult(
                name=suppressor_key,
                processed_audio=audio.copy(),
                metrics=MetricsResult(0, 0, 0, 0, 0),
                success=False,
                error_message=f"Unknown suppressor: {suppressor_key}",
            )
        
        # Make a copy to ensure no modification of original
        audio_copy = audio.copy()
        
        # Run suppressor with timing
        with timer:
            processed = func(audio_copy, sample_rate)
        
        # Ensure output is valid
        if processed is None or len(processed) == 0:
            raise ValueError("Suppressor returned empty output")
        
        # Compute metrics
        metrics = compute_all_metrics(
            audio,
            processed,
            sample_rate,
            processing_time_ms=timer.elapsed_ms,
        )
        
        return SuppressorResult(
            name=suppressor_key,
            processed_audio=processed,
            metrics=metrics,
            success=True,
        )
        
    except Exception as e:
        return SuppressorResult(
            name=suppressor_key,
            processed_audio=audio.copy(),
            metrics=MetricsResult(0, 0, 0, 0, 0),
            success=False,
            error_message=str(e),
        )


def render_sidebar() -> Tuple[Optional[AudioData], List[str]]:
    """Render sidebar with upload and suppressor selection."""
    st.sidebar.title("Audio Noise Lab")
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.subheader("1. Upload Audio")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac"],
        help="Supported formats: WAV, MP3, FLAC",
    )
    
    audio_data = None
    if uploaded_file is not None:
        try:
            with st.sidebar.status("Loading audio...", expanded=False):
                audio_data = load_audio(
                    uploaded_file.getvalue(),
                    target_sr=16000,
                    mono=True,
                    normalize=True,
                )
            
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
            st.sidebar.caption(
                f"Duration: {audio_data.duration_seconds:.2f}s | "
                f"Sample Rate: {audio_data.sample_rate} Hz"
            )
        except Exception as e:
            st.sidebar.error(f"Error loading audio: {e}")
            audio_data = None
    
    st.sidebar.markdown("---")
    
    # Suppressor selection
    st.sidebar.subheader("2. Select Suppressors")
    
    available_suppressors = get_available_suppressors()
    selected = []
    
    # Group by category
    categories = {}
    for key, info in available_suppressors.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, info))
    
    for category, items in categories.items():
        st.sidebar.markdown(f"**{category}**")
        for key, info in items:
            disabled = not info["available"]
            label = info["name"]
            if disabled:
                label += " (unavailable)"
            
            if st.sidebar.checkbox(
                label,
                value=False,
                disabled=disabled,
                help=info.get("install_hint") or info["description"],
                key=f"cb_{key}",
            ):
                selected.append(key)
    
    st.sidebar.markdown("---")
    
    # Run button
    if st.sidebar.button(
        "Run Analysis",
        type="primary",
        disabled=(audio_data is None or len(selected) == 0),
        use_container_width=True,
    ):
        st.session_state["run_analysis"] = True
    else:
        if "run_analysis" not in st.session_state:
            st.session_state["run_analysis"] = False
    
    return audio_data, selected


def render_audio_player(
    audio: np.ndarray,
    sample_rate: int,
    label: str,
) -> None:
    """Render audio player with download button."""
    # Convert to bytes for playback
    audio_bytes = save_audio(audio, sample_rate)
    
    st.audio(audio_bytes, format="audio/wav")
    
    st.download_button(
        label=f"Download {label}",
        data=audio_bytes,
        file_name=f"{label.lower().replace(' ', '_')}.wav",
        mime="audio/wav",
    )


def render_metrics_table(results: Dict[str, SuppressorResult]) -> None:
    """Render metrics comparison table."""
    if not results:
        st.info("No results to display.")
        return
    
    # Build DataFrame
    data = []
    for name, result in results.items():
        if result.success:
            row = {
                "Suppressor": result.name,
                "SNR (dB)": f"{result.metrics.snr_db:.2f}",
                "Seg. SNR (dB)": f"{result.metrics.segmental_snr_db:.2f}",
                "RMSE": f"{result.metrics.rmse:.4f}",
                "LSD": f"{result.metrics.log_spectral_distance:.4f}",
                "Time (ms)": f"{result.metrics.processing_time_ms:.1f}",
            }
        else:
            row = {
                "Suppressor": result.name,
                "SNR (dB)": "Error",
                "Seg. SNR (dB)": "Error",
                "RMSE": "Error",
                "LSD": "Error",
                "Time (ms)": "N/A",
            }
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_detailed_results(
    original: np.ndarray,
    sample_rate: int,
    results: Dict[str, SuppressorResult],
) -> None:
    """Render detailed results for each suppressor."""
    for key, result in results.items():
        suppressor_info = get_available_suppressors().get(key, {})
        display_name = suppressor_info.get("name", key)
        
        with st.expander(f"{display_name}", expanded=False):
            if not result.success:
                st.error(f"Processing failed: {result.error_message}")
                continue
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Audio Playback**")
                render_audio_player(result.processed_audio, sample_rate, display_name)
                
                st.markdown("**Metrics**")
                metrics_dict = result.metrics.to_dict()
                for metric_name, value in metrics_dict.items():
                    st.metric(metric_name, value)
                
                # Interpretation
                interp = interpret_metrics(result.metrics)
                st.markdown("**Interpretation**")
                for key_interp, desc in interp.items():
                    st.caption(f"• {desc}")
            
            with col2:
                st.markdown("**Visual Analysis**")
                fig = plot_full_analysis(
                    original,
                    result.processed_audio,
                    sample_rate,
                    display_name,
                    figsize=(10, 8),
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


def render_main_content(
    audio_data: Optional[AudioData],
    selected_suppressors: List[str],
) -> None:
    """Render main content area with tabs."""
    st.title("Audio Noise Suppression Benchmarking")
    
    if audio_data is None:
        st.info("Upload an audio file using the sidebar to begin.")
        
        # Show system info
        with st.expander("System Information", expanded=True):
            st.markdown("### Available Suppressors")
            
            available = get_available_suppressors()
            for key, info in available.items():
                status = "Available" if info["available"] else "Not Installed"
                icon = "✅" if info["available"] else "❌"
                st.markdown(f"{icon} **{info['name']}** - {status}")
                if info.get("install_hint"):
                    st.code(info["install_hint"], language="bash")
        
        # Show deep learning model status
        with st.expander("Deep Learning Model Status"):
            try:
                from suppressors.deep_model import get_deep_model_status
                status = get_deep_model_status()
                
                st.markdown(f"**PyTorch:** {'Available' if status.torch_available else 'Not Installed'}")
                st.markdown(f"**SpeechBrain:** {'Available' if status.speechbrain_available else 'Not Installed'}")
                st.markdown(f"**Model Loaded:** {'Yes' if status.model_loaded else 'No'}")
                
                if status.model_loaded:
                    st.success(f"Active Model: {status.model_name}")
                elif status.fallback_reason:
                    st.warning(f"Fallback reason: {status.fallback_reason}")
            except Exception as e:
                st.error(f"Could not check deep learning status: {e}")
        
        return
    
    if not selected_suppressors:
        st.warning("Select at least one suppressor from the sidebar.")
        return
    
    if not st.session_state.get("run_analysis", False):
        st.info("Click 'Run Analysis' to process the audio.")
        
        # Show original audio preview
        st.subheader("Original Audio Preview")
        render_audio_player(audio_data.samples, audio_data.sample_rate, "Original")
        return
    
    # Run analysis
    results: Dict[str, SuppressorResult] = {}
    
    progress_bar = st.progress(0, text="Processing...")
    status_text = st.empty()
    
    for i, suppressor_key in enumerate(selected_suppressors):
        suppressor_info = get_available_suppressors().get(suppressor_key, {})
        display_name = suppressor_info.get("name", suppressor_key)
        
        status_text.text(f"Running {display_name}...")
        
        result = run_suppressor(
            suppressor_key,
            audio_data.samples,
            audio_data.sample_rate,
        )
        results[suppressor_key] = result
        
        progress_bar.progress((i + 1) / len(selected_suppressors))
    
    progress_bar.empty()
    status_text.empty()
    
    # Reset run state
    st.session_state["run_analysis"] = False
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Audio Playback",
        "Metrics Table",
        "Visual Analysis",
        "Technical Details",
    ])
    
    with tab1:
        st.subheader("Audio Comparison")
        
        # Original audio
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            render_audio_player(
                audio_data.samples,
                audio_data.sample_rate,
                "Original",
            )
        
        # Processed audio in columns
        successful_results = {k: v for k, v in results.items() if v.success}
        
        if successful_results:
            cols = st.columns(min(len(successful_results), 3))
            for i, (key, result) in enumerate(successful_results.items()):
                suppressor_info = get_available_suppressors().get(key, {})
                display_name = suppressor_info.get("name", key)
                
                with cols[i % 3]:
                    st.markdown(f"**{display_name}**")
                    render_audio_player(
                        result.processed_audio,
                        audio_data.sample_rate,
                        display_name,
                    )
        
        # Show errors if any
        failed_results = {k: v for k, v in results.items() if not v.success}
        if failed_results:
            st.markdown("---")
            st.markdown("**Failed Suppressors**")
            for key, result in failed_results.items():
                st.error(f"{key}: {result.error_message}")
    
    with tab2:
        st.subheader("Metrics Comparison")
        render_metrics_table(results)
        
        # Metrics bar chart
        successful_metrics = {
            get_available_suppressors().get(k, {}).get("name", k): v.metrics
            for k, v in results.items() if v.success
        }
        
        if successful_metrics:
            st.markdown("### Metrics Visualization")
            fig = plot_metrics_comparison(successful_metrics)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    with tab3:
        st.subheader("Visual Analysis")
        
        # Spectrogram comparison
        successful_audio = {
            get_available_suppressors().get(k, {}).get("name", k): v.processed_audio
            for k, v in results.items() if v.success
        }
        
        if successful_audio:
            st.markdown("### Spectrogram Comparison")
            fig = create_comparison_figure(
                audio_data.samples,
                successful_audio,
                audio_data.sample_rate,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Individual analysis
        st.markdown("### Detailed Analysis per Suppressor")
        render_detailed_results(
            audio_data.samples,
            audio_data.sample_rate,
            results,
        )
    
    with tab4:
        st.subheader("Technical Details")
        
        # Audio info
        st.markdown("### Audio Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{audio_data.duration_seconds:.2f} s")
        with col2:
            st.metric("Sample Rate", f"{audio_data.sample_rate} Hz")
        with col3:
            st.metric("Samples", f"{audio_data.n_samples:,}")
        
        # Processing summary
        st.markdown("### Processing Summary")
        summary_data = []
        for key, result in results.items():
            suppressor_info = get_available_suppressors().get(key, {})
            summary_data.append({
                "Suppressor": suppressor_info.get("name", key),
                "Category": suppressor_info.get("category", "Unknown"),
                "Status": "Success" if result.success else "Failed",
                "Processing Time": f"{result.metrics.processing_time_ms:.1f} ms" if result.success else "N/A",
                "Real-time Factor": f"{audio_data.duration_seconds * 1000 / result.metrics.processing_time_ms:.1f}x" if result.success and result.metrics.processing_time_ms > 0 else "N/A",
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Algorithm descriptions
        st.markdown("### Algorithm Descriptions")
        
        descriptions = {
            "spectral_gate": """
            **Spectral Gating** applies a frequency-dependent gain to attenuate
            bins below the estimated noise floor. Uses STFT with overlap-add
            reconstruction.
            """,
            "wiener": """
            **Wiener Filter** computes the optimal linear filter that minimizes
            mean squared error. Uses minimum statistics for noise PSD estimation.
            """,
            "bandpass": """
            **Bandpass Filter** removes frequency content outside the speech band
            (300-3400 Hz). Simple but effective for telephony-quality speech.
            """,
            "noisereduce_stationary": """
            **Noisereduce (Stationary)** uses spectral gating with sophisticated
            noise profiling, assuming noise characteristics are constant.
            """,
            "deep_learning": """
            **Deep Learning** uses SpeechBrain's SepFormer model trained on
            speech enhancement datasets. Provides state-of-the-art performance
            but requires more computation.
            """,
        }
        
        for key in selected_suppressors:
            if key in descriptions:
                suppressor_info = get_available_suppressors().get(key, {})
                st.markdown(f"**{suppressor_info.get('name', key)}**")
                st.markdown(descriptions[key])


def main():
    """Main application entry point."""
    # Initialize session state
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False
    
    # Render sidebar
    audio_data, selected_suppressors = render_sidebar()
    
    # Render main content
    render_main_content(audio_data, selected_suppressors)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Audio Noise Lab v1.0 | "
        "[Documentation](https://github.com/example/audio-noise-lab)"
    )


if __name__ == "__main__":
    main()
