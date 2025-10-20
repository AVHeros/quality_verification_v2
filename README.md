# DVS Quality Verification v2

A comprehensive Python framework for evaluating the quality of synthetic Dynamic Vision Sensor (DVS) event data. This tool provides multiple evaluation pipelines to assess the fidelity of synthetic event streams generated from RGB video sequences.

## 🎯 Overview

Dynamic Vision Sensors (DVS) are bio-inspired cameras that detect changes in brightness asynchronously, producing sparse event streams instead of traditional frames. This project provides robust quality assessment tools for synthetic DVS data generated using tools like [v2e](https://github.com/SensorsINI/v2e).

### Key Features

- **Multi-modal Evaluation**: Compare RGB frames vs DVS frames, RGB frames vs DVS events, and self-quality assessment
- **Comprehensive Metrics**: 19+ quality metrics including MSE, SSIM, PSNR, LPIPS, and specialized event metrics
- **Automated Reporting**: JSON reports with statistical summaries and visualization plots
- **Research-Ready**: Designed for academic research with proper benchmarking against state-of-the-art

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd quality_verification_v2
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

The tool provides three main evaluation pipelines:

#### 1. RGB vs DVS Frames Comparison
Compare original RGB frames with synthesized DVS frames:

```bash
python -m quality_verification.cli rgb-vs-dvs-frames \
    --root /path/to/dataset \
    --output-folder ./results/frames
```

#### 2. RGB vs DVS Events Comparison
Compare RGB frames with DVS event streams (AEDAT format):

```bash
python -m quality_verification.cli rgb-vs-dvs-events \
    --root /path/to/dataset \
    --frame-rate 30.0 \
    --output-folder ./results/events
```

#### 3. DVS Self-Quality Assessment
Evaluate DVS data quality without ground truth:

```bash
python -m quality_verification.cli dvs-self-quality \
    --root /path/to/dataset \
    --window-ms 50.0 \
    --output-folder ./results/self-quality
```

## 📁 Project Structure

```
quality_verification_v2/
├── src/quality_verification/          # Core package
│   ├── cli.py                        # Command-line interface
│   ├── io/                           # Data loading utilities
│   │   ├── event_io.py              # Event stream I/O
│   │   └── frame_io.py              # Frame I/O
│   ├── metrics/                      # Quality metrics
│   │   ├── event_metrics.py         # Event-specific metrics
│   │   ├── frame_metrics.py         # Frame comparison metrics
│   │   └── no_reference.py          # No-reference quality metrics
│   ├── pipelines/                    # Evaluation pipelines
│   │   ├── rgb_vs_dvs_frames.py     # Frame comparison pipeline
│   │   ├── rgb_vs_dvs_events.py     # Event comparison pipeline
│   │   └── dvs_self_quality.py      # Self-quality pipeline
│   └── utils/                        # Utilities
│       ├── plotting.py              # Visualization
│       ├── reporting.py             # Report generation
│       └── path_utils.py            # Path handling
├── dataset/                          # Sample datasets
├── docs/                            # Documentation
│   ├── event_metrics_implementation_guide.csv
│   ├── event_metrics_usage_guide.csv
│   └── v2e_evaluation_pipeline_guide.txt
└── requirements.txt                  # Dependencies
```

## 📊 Supported Metrics

### Frame Comparison Metrics
- **MSE**: Mean Squared Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better, >25 dB is good)
- **SSIM**: Structural Similarity Index (0-1, >0.7 is good)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better, <0.3 is good)

### Event-Specific Metrics
- **Event Density**: Spatial-temporal sparsity of events
- **Event Rate**: Events per second (typical: 10k-100k events/sec)
- **Polarity Accuracy**: ON/OFF event balance
- **Temporal Precision**: Event timing consistency

### No-Reference Quality Metrics
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator
- **NIQE**: Natural Image Quality Evaluator
- **MANIQA**: Transformer-based deep learning quality metric

## 📈 Understanding Results

### Quality Benchmarks

Based on recent research (EVREAL Benchmark 2023):

| Metric | Good | Excellent |
|--------|------|-----------|
| SSIM | >0.70 | >0.80 |
| PSNR | >25 dB | >30 dB |
| LPIPS | <0.30 | <0.20 |
| MSE | <0.05 | <0.02 |

### Output Interpretation

The tool generates:
- **JSON Reports**: Statistical summaries with mean, std, min, max values
- **Visualization Plots**: Time series and histogram plots for each metric
- **Per-pair Analysis**: Individual metric values for each frame/event pair

Example output structure:
```json
{
  "root_path": "/path/to/dataset",
  "pair_count": 88,
  "metrics_summary": {
    "mse": {"mean": 0.062, "std": 0.045, "min": 0.012, "max": 0.234},
    "ssim": {"mean": 0.627, "std": 0.156, "min": 0.234, "max": 0.891},
    "psnr": {"mean": 12.15, "std": 3.42, "min": 6.31, "max": 19.23},
    "lpips": {"mean": 0.731, "std": 0.123, "min": 0.456, "max": 0.934}
  }
}
```

## 🔧 Advanced Usage

### Custom Metric Selection
```bash
python -m quality_verification.cli rgb-vs-dvs-frames \
    --root /path/to/dataset \
    --metrics mse ssim psnr \
    --limit 100 \
    --output-folder ./results
```

### Event Stream Analysis with Temporal Offset
```bash
python -m quality_verification.cli rgb-vs-dvs-events \
    --root /path/to/dataset \
    --frame-rate 30.0 \
    --sync-offset-us 1000 \
    --limit 50 \
    --output-folder ./results
```

### Self-Quality with Custom Window Size
```bash
python -m quality_verification.cli dvs-self-quality \
    --root /path/to/dataset \
    --window-ms 100.0 \
    --limit 200 \
    --output-folder ./results
```

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- NumPy
- OpenCV
- scikit-image
- SciPy
- PyTorch
- torchvision

### Specialized Libraries
- **lpips**: Perceptual similarity metrics
- **dv-processing**: DVS event stream processing
- **pyiqa**: Image quality assessment
- **matplotlib**: Visualization
- **tqdm**: Progress bars

## 🗂️ Dataset Format

The tool expects datasets with the following structure:

```
dataset_root/
├── rgb_full/                    # Original RGB frames
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── dvs_full/                    # Synthesized DVS frames
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── events.aedat4               # Event stream (for event analysis)
```

## 🔬 Research Context

This tool is designed for researchers working on:
- **Event Camera Simulation**: Validating synthetic event data quality
- **Computer Vision**: Comparing traditional vs event-based vision
- **Robotics**: Evaluating event cameras for autonomous systems
- **Neuromorphic Computing**: Assessing bio-inspired sensor data

### Citation

If you use this tool in your research, please cite:

```bibtex
@software{dvs_quality_verification_v2,
  title={},
  author={},
  year={},
  url={[Repository URL]}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 📚 Documentation

- **Implementation Guide**: `docs/event_metrics_implementation_guide.csv`
- **Usage Guide**: `docs/event_metrics_usage_guide.csv`
- **V2E Pipeline**: `docs/v2e_evaluation_pipeline_guide.txt`

## 🐛 Troubleshooting

### Common Issues

1. **AEDAT file loading errors**: Ensure `dv-processing` is properly installed
2. **CUDA out of memory**: Reduce batch size or use `--limit` parameter
3. **Missing plots**: Check that `matplotlib` backend is properly configured

### Performance Tips

- Use `--limit` parameter for large datasets during development
- Enable GPU acceleration for LPIPS calculations
- Process datasets in chunks for memory efficiency

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [v2e](https://github.com/SensorsINI/v2e) for DVS simulation
- [dv-processing](https://gitlab.com/inivation/dv/dv-processing) for event stream handling
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for perceptual metrics
- The event camera research community for valuable insights

---

For questions, issues, or contributions, please visit our [GitHub repository](repository-url) or contact [].