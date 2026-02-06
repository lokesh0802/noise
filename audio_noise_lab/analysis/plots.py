"""
Audio Analysis Visualization.

Creates publication-quality plots for comparing audio signals
using matplotlib only (no seaborn dependency).

Plots include:
- Waveform overlay comparison
- Spectrogram (dB scale)
- Difference spectrogram
- Combined analysis figure

All plots use consistent styling suitable for technical documents.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List
from scipy import signal as scipy_signal
import io


# Configure matplotlib for consistent styling
plt.style.use('default')
COLORS = {
    'original': '#2196F3',      # Blue
    'processed': '#4CAF50',     # Green
    'difference': '#FF5722',    # Orange
    'background': '#FAFAFA',
    'grid': '#E0E0E0',
}


def _compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram in dB scale.
    
    Returns frequencies, times, and magnitude spectrogram in dB.
    """
    f, t, Sxx = scipy_signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling='spectrum',
    )
    
    # Convert to dB with floor
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    return f, t, Sxx_db


def plot_waveform_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    title: str = "Waveform Comparison",
    figsize: Tuple[int, int] = (12, 4),
) -> Figure:
    """
    Create waveform overlay plot comparing original and processed audio.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Create time axis
    time = np.arange(min_len) / sample_rate
    
    # Plot with transparency for overlay
    ax.plot(time, original, color=COLORS['original'], alpha=0.7,
            linewidth=0.5, label='Original')
    ax.plot(time, processed, color=COLORS['processed'], alpha=0.7,
            linewidth=0.5, label='Processed')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    n_fft: int = 2048,
    hop_length: int = 512,
    vmin: float = -80,
    vmax: float = 0,
    cmap: str = 'magma',
) -> Figure:
    """
    Create spectrogram plot in dB scale.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    figsize : tuple
        Figure size
    n_fft : int
        FFT size
    hop_length : int
        Hop length
    vmin, vmax : float
        Color scale range in dB
    cmap : str
        Colormap name
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Compute spectrogram
    f, t, Sxx_db = _compute_spectrogram(audio, sample_rate, n_fft, hop_length)
    
    # Normalize to 0 dB maximum
    Sxx_db = Sxx_db - np.max(Sxx_db)
    
    # Plot
    im = ax.pcolormesh(t, f / 1000, Sxx_db, shading='gouraud',
                       cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (kHz)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, sample_rate / 2000)  # Up to Nyquist in kHz
    
    plt.tight_layout()
    return fig


def plot_difference_spectrogram(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    title: str = "Difference Spectrogram (Removed Noise)",
    figsize: Tuple[int, int] = (12, 4),
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Figure:
    """
    Create difference spectrogram showing what was removed.
    
    Useful for visualizing which frequency components were
    attenuated by the noise suppression algorithm.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
    sample_rate : int
        Sample rate in Hz
    title : str
        Plot title
    figsize : tuple
        Figure size
    n_fft : int
        FFT size
    hop_length : int
        Hop length
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    # Compute difference (what was removed)
    difference = original - processed
    
    # Compute spectrogram of difference
    f, t, Sxx_db = _compute_spectrogram(difference, sample_rate, n_fft, hop_length)
    
    # Plot with diverging colormap
    im = ax.pcolormesh(t, f / 1000, Sxx_db, shading='gouraud',
                       cmap='RdBu_r', vmin=-60, vmax=0)
    
    cbar = fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (kHz)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, sample_rate / 2000)
    
    plt.tight_layout()
    return fig


def plot_full_analysis(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    suppressor_name: str = "Processed",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Create comprehensive analysis figure with multiple plots.
    
    Includes:
    - Waveform comparison
    - Original spectrogram
    - Processed spectrogram
    - Difference spectrogram
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    processed : np.ndarray
        Processed audio signal
    sample_rate : int
        Sample rate in Hz
    suppressor_name : str
        Name of the suppressor for labels
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=COLORS['background'])
    
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    time = np.arange(min_len) / sample_rate
    
    # 1. Waveform comparison (top left)
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLORS['background'])
    ax1.plot(time, original, color=COLORS['original'], alpha=0.7,
             linewidth=0.5, label='Original')
    ax1.plot(time, processed, color=COLORS['processed'], alpha=0.7,
             linewidth=0.5, label=suppressor_name)
    ax1.set_xlabel('Time (s)', fontsize=9)
    ax1.set_ylabel('Amplitude', fontsize=9)
    ax1.set_title('Waveform Comparison', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time[-1])
    
    # 2. Original spectrogram (top right)
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLORS['background'])
    f, t, Sxx_orig = _compute_spectrogram(original, sample_rate)
    Sxx_orig = Sxx_orig - np.max(Sxx_orig)
    im2 = ax2.pcolormesh(t, f / 1000, Sxx_orig, shading='gouraud',
                         cmap='magma', vmin=-80, vmax=0)
    fig.colorbar(im2, ax=ax2, label='dB')
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.set_ylabel('Frequency (kHz)', fontsize=9)
    ax2.set_title('Original Spectrogram', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, sample_rate / 2000)
    
    # 3. Processed spectrogram (bottom left)
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLORS['background'])
    f, t, Sxx_proc = _compute_spectrogram(processed, sample_rate)
    Sxx_proc = Sxx_proc - np.max(Sxx_proc)
    im3 = ax3.pcolormesh(t, f / 1000, Sxx_proc, shading='gouraud',
                         cmap='magma', vmin=-80, vmax=0)
    fig.colorbar(im3, ax=ax3, label='dB')
    ax3.set_xlabel('Time (s)', fontsize=9)
    ax3.set_ylabel('Frequency (kHz)', fontsize=9)
    ax3.set_title(f'{suppressor_name} Spectrogram', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, sample_rate / 2000)
    
    # 4. Difference spectrogram (bottom right)
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLORS['background'])
    difference = original - processed
    f, t, Sxx_diff = _compute_spectrogram(difference, sample_rate)
    im4 = ax4.pcolormesh(t, f / 1000, Sxx_diff, shading='gouraud',
                         cmap='RdBu_r', vmin=-60, vmax=0)
    fig.colorbar(im4, ax=ax4, label='dB')
    ax4.set_xlabel('Time (s)', fontsize=9)
    ax4.set_ylabel('Frequency (kHz)', fontsize=9)
    ax4.set_title('Removed Noise Spectrogram', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, sample_rate / 2000)
    
    plt.tight_layout()
    return fig


def create_comparison_figure(
    original: np.ndarray,
    results: dict,
    sample_rate: int,
    figsize: Tuple[int, int] = (16, 3),
) -> Figure:
    """
    Create a comparison figure showing all suppressors side by side.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio signal
    results : dict
        Dictionary mapping suppressor names to processed audio
    sample_rate : int
        Sample rate in Hz
    figsize : tuple
        Figure size per row
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    n_suppressors = len(results)
    if n_suppressors == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
        return fig
    
    # Create figure with one column per suppressor + original
    n_cols = n_suppressors + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0], figsize[1]),
                             facecolor=COLORS['background'])
    
    if n_cols == 1:
        axes = [axes]
    
    # Plot original spectrogram
    f, t, Sxx = _compute_spectrogram(original, sample_rate)
    Sxx = Sxx - np.max(Sxx)
    
    axes[0].pcolormesh(t, f / 1000, Sxx, shading='gouraud',
                       cmap='magma', vmin=-80, vmax=0)
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Freq (kHz)', fontsize=9)
    axes[0].set_xlabel('Time (s)', fontsize=9)
    axes[0].set_ylim(0, sample_rate / 2000)
    
    # Plot each suppressor result
    for i, (name, processed) in enumerate(results.items(), 1):
        min_len = min(len(original), len(processed))
        f, t, Sxx = _compute_spectrogram(processed[:min_len], sample_rate)
        Sxx = Sxx - np.max(Sxx)
        
        axes[i].pcolormesh(t, f / 1000, Sxx, shading='gouraud',
                           cmap='magma', vmin=-80, vmax=0)
        axes[i].set_title(name, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=9)
        axes[i].set_ylim(0, sample_rate / 2000)
        axes[i].set_yticklabels([])
    
    plt.tight_layout()
    return fig


def figure_to_bytes(fig: Figure, format: str = 'png', dpi: int = 150) -> bytes:
    """
    Convert matplotlib figure to bytes for Streamlit display.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    format : str
        Output format (png, svg, pdf)
    dpi : int
        Resolution in dots per inch
        
    Returns
    -------
    bytes
        Image data
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def plot_metrics_comparison(
    metrics_dict: dict,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Create bar chart comparing metrics across suppressors.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping suppressor names to MetricsResult objects
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not metrics_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No metrics to display', ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=COLORS['background'])
    
    names = list(metrics_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # Extract metric values
    snr_values = [m.snr_db for m in metrics_dict.values()]
    seg_snr_values = [m.segmental_snr_db for m in metrics_dict.values()]
    rmse_values = [m.rmse for m in metrics_dict.values()]
    lsd_values = [m.log_spectral_distance for m in metrics_dict.values()]
    
    x = np.arange(len(names))
    
    # SNR plot
    axes[0, 0].bar(x, snr_values, color=colors)
    axes[0, 0].set_title('SNR (dB)', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Segmental SNR plot
    axes[0, 1].bar(x, seg_snr_values, color=colors)
    axes[0, 1].set_title('Segmental SNR (dB)', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # RMSE plot
    axes[1, 0].bar(x, rmse_values, color=colors)
    axes[1, 0].set_title('RMSE', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # LSD plot
    axes[1, 1].bar(x, lsd_values, color=colors)
    axes[1, 1].set_title('Log Spectral Distance', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
