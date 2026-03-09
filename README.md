# Interpretable NLP Models

A comprehensive framework for training and interpreting BERT-based text classification models using multiple explanation methods including SHAP, LIME, attention visualization, and integrated gradients.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational tool for Explainable AI (XAI).**

- XAI outputs may be unstable or misleading
- Explanations are not a substitute for human judgment
- **DO NOT** use for regulated decisions without human review
- Results may vary across different runs and configurations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Interpretable-NLP-Models.git
cd Interpretable-NLP-Models

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train a model with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/default.yaml --experiment_name my_experiment

# Use synthetic dataset for faster testing
python scripts/train.py --use_synthetic --experiment_name synthetic_test
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## 📁 Project Structure

```
interpretable-nlp-models/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/             # Model implementations
│   ├── explainers/         # Explanation methods
│   ├── eval/               # Evaluation metrics
│   ├── viz/                # Visualization tools
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── demo/                   # Streamlit demo application
├── tests/                  # Unit tests
├── assets/                 # Generated visualizations and outputs
├── data/                   # Data storage
└── outputs/                # Training outputs and checkpoints
```

## Features

### Model Architecture
- **BERT-based text classification** with customizable architecture
- **Device fallback**: CUDA → MPS (Apple Silicon) → CPU
- **Deterministic training** with proper seeding
- **Flexible configuration** via YAML files

### Explanation Methods
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local linear approximations
- **Attention Visualization**: BERT attention weight analysis
- **Integrated Gradients**: Gradient-based attribution
- **Saliency Maps**: Gradient-based saliency

### Evaluation Metrics
- **Faithfulness**: Deletion/insertion tests, sufficiency/necessity
- **Stability**: Cross-run consistency, explanation similarity
- **Fidelity**: Surrogate model accuracy
- **Calibration**: Expected Calibration Error (ECE), Brier Score
- **Robustness**: Adversarial robustness, noise sensitivity

### Visualization
- **Interactive plots** with Plotly
- **Static visualizations** with Matplotlib/Seaborn
- **Comparison plots** across explanation methods
- **Attention heatmaps** and saliency maps
- **Performance dashboards**

## Datasets

### 20 Newsgroups
- **Description**: Text classification dataset with 20 categories
- **Size**: ~18,000 documents
- **Classes**: 20 newsgroup categories
- **Usage**: `load_20newsgroups()`

### Synthetic Dataset
- **Description**: Generated text classification dataset
- **Size**: Configurable (default: 1,000 samples)
- **Classes**: Configurable (default: 5 classes)
- **Usage**: `create_synthetic_dataset()`

## 🔧 Configuration

### Model Configuration
```yaml
model:
  model_name: "bert-base-uncased"  # BERT model variant
  num_labels: 20                   # Number of classes
  dropout_rate: 0.1               # Dropout rate
  freeze_bert: false              # Freeze BERT parameters
```

### Training Configuration
```yaml
training:
  epochs: 3                       # Training epochs
  batch_size: 16                  # Batch size
  max_length: 512                 # Maximum sequence length
  learning_rate: 2e-5             # Learning rate
  weight_decay: 0.01              # Weight decay
  warmup_steps: 500               # Warmup steps
```

### Explanation Configuration
```yaml
explanation:
  use_shap: true                  # Enable SHAP explanations
  use_lime: true                  # Enable LIME explanations
  use_attention: true             # Enable attention visualization
  use_ig: true                    # Enable integrated gradients
  shap_max_samples: 100           # SHAP sampling budget
  lime_num_features: 10           # LIME feature count
  attention_layer: -1             # BERT layer for attention
  ig_n_steps: 50                  # IG integration steps
```

## Usage Examples

### Training a Model

```python
from src.data.loader import load_20newsgroups
from src.models.bert import InterpretableBERT
from src.explainers.nlp_explainer import NLPExplainer

# Load data
dataset = load_20newsgroups()
train_dataset, val_dataset, test_dataset = dataset.split()

# Create model
model = InterpretableBERT(
    model_name="bert-base-uncased",
    num_labels=20,
)

# Train model (simplified)
# ... training code ...

# Generate explanations
explainer = NLPExplainer(model)
explanations = explainer.compare_explanations(
    text="Your input text here",
    methods=["shap", "lime", "attention"]
)
```

### Evaluating Model Performance

```python
from src.eval.metrics import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(
    model,
    test_dataset.texts,
    test_dataset.labels,
    test_dataset.label_names,
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Creating Visualizations

```python
from src.viz.visualizer import NLPVisualizer

visualizer = NLPVisualizer(model.tokenizer)

# Plot SHAP values
shap_fig = visualizer.plot_shap_values(text, shap_values)

# Plot attention heatmap
attention_fig = visualizer.plot_attention_heatmap(
    text, attention_weights, tokens
)

# Compare explanations
comparison_fig = visualizer.plot_explanation_comparison(
    text, explanations, methods=["shap", "lime", "attention"]
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## Development

### Code Formatting
```bash
# Format code with black
black src/ scripts/ demo/

# Lint with ruff
ruff check src/ scripts/ demo/

# Run pre-commit hooks
pre-commit run --all-files
```

### Adding New Explanation Methods

1. Create a new method in `src/explainers/nlp_explainer.py`
2. Add visualization support in `src/viz/visualizer.py`
3. Update evaluation metrics in `src/eval/metrics.py`
4. Add tests in `tests/`

## Research Applications

This framework is designed for:
- **Research**: Comparing explanation methods
- **Education**: Learning XAI concepts
- **Prototyping**: Rapid experimentation
- **Benchmarking**: Evaluating explanation quality

## ⚠️ Limitations

- **Computational Cost**: Some explanation methods are computationally expensive
- **Stability**: Explanations may vary across runs
- **Interpretability**: Not all explanations are equally reliable
- **Domain Specificity**: Results may not generalize across domains
- **Human Factors**: Explanation quality depends on user expertise

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
- Lundberg, S. M., & Lee, S. I. "A unified approach to interpreting model predictions." NeurIPS 2017.
- Ribeiro, M. T., Singh, S., & Guestrin, C. "Why should I trust you?" KDD 2016.
- Sundararajan, M., et al. "Axiomatic attribution for deep networks." ICML 2017.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Remember**: This tool is for research and education. Always validate explanations with domain experts before making important decisions.
# Interpretable-NLP-Models
