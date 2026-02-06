# Audio Noise Lab

A professional-grade benchmarking system for audio noise suppression algorithms. Compare classical signal processing methods with modern deep learning approaches using research-standard metrics and visualizations.

## Overview

Audio Noise Lab provides a unified framework for:
- Evaluating multiple noise suppression algorithms on identical input
- Computing research-grade audio quality metrics
- Visualizing spectral changes and noise reduction effectiveness
- Comparing classical DSP methods with neural network approaches

This is a research tool, not a demo. Every algorithm is fully implemented with proper DSP mathematics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Installation

### Core Installation

```bash
# Clone and enter directory
cd audio_noise_lab

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt
```

### Optional: Deep Learning Support

For SpeechBrain (recommended for speech enhancement):
```bash
pip install torch speechbrain
```

For Demucs (better for music/voice separation):
```bash
pip install demucs
```

### Optional: MP3 Support

MP3 loading requires FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Noise Suppression Algorithms

### 1. Spectral Gating

**Theory:** STFT-based spectral gating estimates the noise floor from the signal statistics and attenuates frequency bins below a threshold.

**Implementation:**
- Computes Short-Time Fourier Transform (STFT) with Hann window
- Estimates noise floor using percentile statistics across time frames
- Generates a soft gain mask with attack/release smoothing
- Applies overlap-add reconstruction to prevent discontinuities

**When to use:**
- Stationary background noise (fan, air conditioning)
- When computational efficiency is important
- As a baseline for comparing other methods

**Parameters:**
- `n_fft`: FFT size (default: 2048) - larger = better frequency resolution
- `threshold_db`: Gate threshold above noise floor (default: -12 dB)
- `reduction_db`: Attenuation when gated (default: -24 dB)

**Limitations:**
- Can introduce "musical noise" artifacts if threshold is too aggressive
- Poor performance on non-stationary noise
- May attenuate quiet speech segments

### 2. Wiener Filter

**Theory:** The Wiener filter is the optimal linear filter for signal estimation in additive noise, minimizing mean squared error.

For noisy signal Y = S + N:
```
H(f) = |S(f)|² / (|S(f)|² + |N(f)|²) = SNR(f) / (1 + SNR(f))
```

**Implementation:**
- Estimates noise PSD using minimum statistics (Martin, 2001)
- Computes a priori SNR using decision-directed approach (Ephraim & Malah, 1984)
- Applies temporal smoothing to reduce musical noise
- Preserves phase information from original signal

**When to use:**
- When optimal MSE performance is needed
- Moderate to high SNR conditions
- As a reference for classical methods

**Parameters:**
- `alpha`: Decision-directed smoothing factor (default: 0.98)
- `gain_floor`: Minimum gain to prevent complete suppression (default: 0.1)

**Limitations:**
- Requires accurate noise PSD estimation
- Performance degrades at low SNR
- Assumes noise and signal are uncorrelated

### 3. Bandpass Filter

**Theory:** Frequency filtering removes out-of-band noise by passing only frequencies within a specified range.

**Implementation:**
- Uses Butterworth IIR filter design for maximally flat passband
- Applies zero-phase filtering (forward-backward) to eliminate phase distortion
- Default band (300-3400 Hz) targets the speech intelligibility range

**When to use:**
- Known out-of-band interference (rumble, hiss)
- Pre-processing before other methods
- Telephony-grade speech processing

**Voice Band Frequencies:**
- Below 300 Hz: Rumble, handling noise, HVAC
- 300-3400 Hz: Speech intelligibility range
- Above 3400 Hz: Sibilants, high-frequency noise

**Limitations:**
- Does not remove in-band noise
- May affect naturalness of wideband speech
- Cannot adapt to signal content

### 4. Noisereduce Library

**Theory:** The noisereduce library implements sophisticated spectral gating with adaptive noise profiling.

**Stationary Mode:**
- Estimates noise profile from the entire signal or a provided clip
- Assumes noise characteristics remain constant
- Good for fans, air conditioning, electrical hum

**Non-stationary Mode:**
- Uses running statistics for noise estimation
- Adapts to time-varying noise
- Better for traffic, crowds, environmental noise

**When to use:**
- General-purpose noise reduction
- When manual parameter tuning is undesirable
- Audio production workflows

**Limitations:**
- Black-box implementation limits interpretability
- May over-smooth transients
- Computational overhead for long audio

### 5. Deep Learning (SpeechBrain SepFormer)

**Theory:** Neural networks learn to separate clean speech from noise through supervised training on large datasets.

**SepFormer Architecture:**
- Dual-path architecture with self-attention
- Directly processes waveforms (no explicit STFT)
- Trained on WHAMR! dataset (noisy reverberant speech)

**When to use:**
- Non-stationary and complex noise
- When computational resources are available
- State-of-the-art performance is required

**Limitations:**
- Requires PyTorch and model download (~100MB)
- Slower than classical methods (especially on CPU)
- May introduce artifacts on out-of-distribution audio
- Fixed 8kHz sample rate (internal resampling applied)

### 6. RNNoise (Xiph.org)

**Theory:** RNNoise uses a hybrid approach combining classical DSP with recurrent neural networks for real-time noise suppression.

**Architecture:**
- **Feature Extraction:** 22 Bark-scale frequency bands + pitch features
- **GRU Network:** Three GRU layers (24, 48, 96 units) for temporal context
- **Gain Prediction:** Network outputs per-band gains (0 to 1)
- **Pitch Filtering:** Comb filter for pitch period enhancement

**Key Advantages:**
- Extremely low latency (~10ms frame size)
- Minimal CPU usage (designed for real-time VoIP)
- No voice activity detection required
- Works well across many noise types

**When to use:**
- Real-time applications (VoIP, streaming)
- When latency is critical
- Resource-constrained environments
- General-purpose noise suppression

**Installation:**

RNNoise requires building from source as no reliable PyPI packages exist:

```bash
# macOS/Linux
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make
sudo make install

# Then install Python bindings (if available)
pip install rnnoise
```

For Python bindings, you may need to build a wrapper or use ctypes to interface
with the compiled library. See the RNNoise repository for details.

**Limitations:**
- Requires native library compilation (platform-dependent)
- May not be available on all systems (especially Windows)
- Fixed 48kHz internal processing rate
- Less effective on very low SNR signals
- Cannot handle music (speech-only training)

**Reference:** Valin, J.M. (2018). "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement." https://arxiv.org/abs/1709.08243

## Audio Quality Metrics

### SNR (Signal-to-Noise Ratio)

**Definition:**
```
SNR = 10 × log₁₀(P_signal / P_noise) dB
```

**Interpretation:**
- Measures ratio of signal power to removed noise power
- Higher values indicate more aggressive noise removal
- Very high SNR may indicate signal distortion

**Typical Values:**
- < 0 dB: Minimal processing
- 0-10 dB: Light noise reduction
- 10-20 dB: Moderate noise reduction
- > 20 dB: Aggressive noise reduction

### Segmental SNR

**Definition:**
```
SNR_seg = (1/N) × Σ SNR_frame[i]
```

Each frame SNR is clipped to [-10, 35] dB to prevent outliers.

**Interpretation:**
- Better correlates with perceived quality than global SNR
- Accounts for time-varying signal characteristics
- Standard metric in speech enhancement literature

### RMSE (Root Mean Square Error)

**Definition:**
```
RMSE = √(mean((original - processed)²))
```

**Interpretation:**
- Measures average sample-level difference
- Lower values indicate less modification
- 0 means identical signals

**Typical Values (normalized audio):**
- < 0.01: Minimal change
- 0.01-0.05: Light modification
- 0.05-0.1: Moderate modification
- > 0.1: Heavy modification

### Log Spectral Distance (LSD)

**Definition:**
```
LSD = √(mean((10×log₁₀(S₁) - 10×log₁₀(S₂))²)) dB
```

**Interpretation:**
- Measures spectral similarity in log domain
- Better reflects perceptual differences than linear metrics
- Lower is better

**Typical Values:**
- < 1 dB: Excellent spectral preservation
- 1-2 dB: Good quality
- 2-4 dB: Moderate distortion
- > 4 dB: Significant spectral change

## Metric Interpretation Guidelines

**Ideal Noise Reduction:**
- Moderate SNR improvement (10-20 dB)
- Low RMSE (signal preserved)
- Low LSD (spectral characteristics maintained)
- Fast processing time

**Over-aggressive Processing:**
- Very high SNR (> 25 dB)
- High RMSE (> 0.1)
- High LSD (> 4 dB)
- May sound "underwater" or have musical noise

**Under-processing:**
- Very low SNR (< 5 dB)
- Very low RMSE (< 0.01)
- Noise still audible

## Project Structure

```
audio_noise_lab/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── suppressors/              # Noise suppression algorithms
│   ├── __init__.py
│   ├── spectral_gate.py      # Manual STFT-based gating
│   ├── wiener.py             # Frequency-domain Wiener filter
│   ├── bandpass.py           # IIR bandpass/highpass filters
│   ├── noisereduce_wrapper.py # noisereduce library interface
│   └── deep_model.py         # Deep learning models
├── analysis/                 # Metrics and visualization
│   ├── __init__.py
│   ├── metrics.py            # Audio quality metrics
│   └── plots.py              # Matplotlib visualizations
└── utils/                    # Utilities
    ├── __init__.py
    ├── audio_io.py           # Audio loading/saving
    └── timing.py             # Performance measurement
```

## API Usage

### Standalone Suppressor Usage

```python
import numpy as np
from utils.audio_io import load_audio
from suppressors.spectral_gate import spectral_gate
from suppressors.wiener import wiener_filter

# Load audio
audio_data = load_audio("noisy_speech.wav", target_sr=16000)

# Apply spectral gating
denoised_sg = spectral_gate(
    audio_data.samples, 
    audio_data.sample_rate,
    threshold_db=-12.0,
    reduction_db=-24.0,
)

# Apply Wiener filter
denoised_wiener = wiener_filter(
    audio_data.samples,
    audio_data.sample_rate,
    gain_floor=0.1,
)
```

### Computing Metrics

```python
from analysis.metrics import compute_all_metrics, interpret_metrics

metrics = compute_all_metrics(
    original=audio_data.samples,
    processed=denoised_sg,
    sample_rate=audio_data.sample_rate,
    processing_time_ms=50.0,
)

print(f"SNR: {metrics.snr_db:.2f} dB")
print(f"LSD: {metrics.log_spectral_distance:.4f}")

# Get human-readable interpretation
interp = interpret_metrics(metrics)
for key, desc in interp.items():
    print(f"{key}: {desc}")
```

### Creating Plots

```python
from analysis.plots import plot_full_analysis
import matplotlib.pyplot as plt

fig = plot_full_analysis(
    original=audio_data.samples,
    processed=denoised_sg,
    sample_rate=audio_data.sample_rate,
    suppressor_name="Spectral Gate",
)
plt.savefig("analysis.png", dpi=150)
```

## Known Limitations

1. **Sample Rate:** All processing internally uses 16 kHz. Higher sample rates are downsampled.

2. **Mono Processing:** Stereo files are converted to mono. Channel preservation not implemented.

3. **Memory:** Processing very long files (> 10 minutes) may require significant RAM.

4. **Real-time:** This is a batch processing tool, not designed for real-time streaming.

5. **Deep Learning:**
   - SpeechBrain model is optimized for speech, not music/general audio
   - First run downloads ~100MB model weights
   - CPU inference is significantly slower than GPU

6. **Platform-specific:**
   - MP3 support requires FFmpeg installation
   - Some deep learning operations may not work on Apple Silicon without Rosetta
   - RNNoise requires native library compilation (may not work on Windows)

## References

### Classical Methods
- Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE TASSP*
- Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator. *IEEE TASSP*
- Martin, R. (2001). Noise power spectral density estimation based on optimal smoothing and minimum statistics. *IEEE TASSP*

### Deep Learning
- Subakan, C., et al. (2021). Attention is all you need in speech separation. *ICASSP*
- Valin, J.M. (2018). A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. *arXiv:1709.08243*
- SpeechBrain: A General-Purpose Speech Toolkit. https://speechbrain.github.io/
- RNNoise: https://github.com/xiph/rnnoise

### Metrics
- ITU-T P.862 (2001). Perceptual evaluation of speech quality (PESQ)
- Hu, Y., & Loizou, P. (2008). Evaluation of objective quality measures for speech enhancement. *IEEE TASLP*

## License

MIT License. See LICENSE file for details.

## Contributing

Contributions welcome. Please ensure:
- All functions have type hints
- New suppressors follow the pure function pattern
- Metrics are properly documented
- Tests pass before submitting PR
