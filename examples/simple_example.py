"""Simple example script demonstrating interpretable NLP models."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.loader import create_synthetic_dataset
from src.models.bert import InterpretableBERT
from src.explainers.nlp_explainer import NLPExplainer
from src.viz.visualizer import NLPVisualizer
from src.utils.device import set_seed, get_device


def main():
    """Run a simple example."""
    print("🧠 Interpretable NLP Models - Simple Example")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_samples=100,
        n_classes=3,
        random_state=42,
    )
    
    print(f"Dataset size: {len(dataset.texts)} samples")
    print(f"Number of classes: {dataset.n_classes}")
    print(f"Class names: {dataset.label_names}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        test_size=0.3,
        val_size=0.2,
        random_state=42,
    )
    
    print(f"Train: {len(train_dataset.texts)}, Val: {len(val_dataset.texts)}, Test: {len(test_dataset.texts)}")
    
    # Create model
    print("\n🤖 Creating BERT model...")
    model = InterpretableBERT(
        model_name="bert-base-uncased",
        num_labels=dataset.n_classes,
        dropout_rate=0.1,
    )
    
    print(f"Model created with {model.num_labels} output classes")
    
    # Make predictions on test set
    print("\n🔮 Making predictions...")
    predictions = model.predict(test_dataset.texts)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_dataset.labels)
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Show some predictions
    print("\n📝 Sample predictions:")
    for i in range(min(5, len(test_dataset.texts))):
        text = test_dataset.texts[i][:50] + "..." if len(test_dataset.texts[i]) > 50 else test_dataset.texts[i]
        pred_class = dataset.label_names[predictions[i]]
        true_class = dataset.label_names[test_dataset.labels[i]]
        print(f"Text: {text}")
        print(f"Predicted: {pred_class}, True: {true_class}")
        print()
    
    # Generate explanations for a sample
    print("🔍 Generating explanations...")
    sample_text = test_dataset.texts[0]
    print(f"Sample text: {sample_text}")
    
    # Create explainer
    explainer = NLPExplainer(model)
    
    # Generate different types of explanations
    explanations = {}
    
    try:
        # SHAP explanation
        print("  - SHAP explanation...")
        shap_exp = explainer.explain_with_shap([sample_text], method="kernel", max_samples=50)
        explanations["shap"] = shap_exp
        print("    ✓ SHAP completed")
    except Exception as e:
        print(f"    ✗ SHAP failed: {e}")
    
    try:
        # LIME explanation
        print("  - LIME explanation...")
        lime_exp = explainer.explain_with_lime([sample_text], class_names=dataset.label_names)
        explanations["lime"] = lime_exp
        print("    ✓ LIME completed")
    except Exception as e:
        print(f"    ✗ LIME failed: {e}")
    
    try:
        # Attention explanation
        print("  - Attention explanation...")
        att_exp = explainer.explain_with_attention([sample_text])
        explanations["attention"] = att_exp
        print("    ✓ Attention completed")
    except Exception as e:
        print(f"    ✗ Attention failed: {e}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = NLPVisualizer(model.tokenizer)
    
    try:
        # Create comparison plot
        comparison_fig = visualizer.plot_explanation_comparison(
            sample_text,
            explanations,
            methods=["shap", "lime", "attention"],
        )
        
        # Save plot
        os.makedirs("assets", exist_ok=True)
        comparison_fig.savefig("assets/explanation_comparison.png", bbox_inches="tight", dpi=150)
        print("✓ Comparison plot saved to assets/explanation_comparison.png")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
    
    # Model evaluation
    print("\n📈 Model evaluation...")
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
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    print("\n✅ Example completed successfully!")
    print("\nTo explore more features:")
    print("1. Run the Streamlit demo: streamlit run demo/app.py")
    print("2. Train a full model: python scripts/train.py --use_synthetic")
    print("3. Check the README.md for more details")


if __name__ == "__main__":
    main()
