"""Main training script for Interpretable NLP Models."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import yaml

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import NLPDataset, load_20newsgroups, create_synthetic_dataset
from src.models.bert import InterpretableBERT, TextClassificationDataset
from src.explainers.nlp_explainer import NLPExplainer
from src.eval.metrics import ModelEvaluator
from src.viz.visualizer import NLPVisualizer
from src.utils.device import get_device, set_seed, create_output_dir


def train_model(
    model: InterpretableBERT,
    train_dataset: NLPDataset,
    val_dataset: NLPDataset,
    config: Dict[str, Any],
    output_dir: str,
) -> InterpretableBERT:
    """Train the BERT model.
    
    Args:
        model: The model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration.
        output_dir: Output directory for checkpoints.
        
    Returns:
        Trained model.
    """
    # Create PyTorch datasets
    train_pytorch_dataset = TextClassificationDataset(
        train_dataset.texts,
        train_dataset.labels,
        model.tokenizer,
        max_length=config.get("max_length", 512),
    )
    
    val_pytorch_dataset = TextClassificationDataset(
        val_dataset.texts,
        val_dataset.labels,
        model.tokenizer,
        max_length=config.get("max_length", 512),
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 16),
        warmup_steps=config.get("warmup_steps", 500),
        weight_decay=config.get("weight_decay", 0.01),
        logging_dir=f"{output_dir}/logs",
        logging_steps=config.get("logging_steps", 100),
        evaluation_strategy="steps",
        eval_steps=config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Custom trainer class for our model
    class CustomTrainer(Trainer):
        def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
            super().__init__(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pytorch_dataset,
        eval_dataset=val_pytorch_dataset,
        tokenizer=model.tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    model.save_model(os.path.join(output_dir, "final_model"))
    
    return model


def evaluate_model(
    model: InterpretableBERT,
    test_dataset: NLPDataset,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Evaluate the trained model.
    
    Args:
        model: The trained model.
        test_dataset: Test dataset.
        config: Evaluation configuration.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary containing evaluation results.
    """
    print("Evaluating model...")
    
    evaluator = ModelEvaluator()
    
    # Evaluate model performance
    performance_metrics = evaluator.evaluate_model(
        model,
        test_dataset.texts,
        test_dataset.labels,
        test_dataset.label_names,
    )
    
    print(f"Test Accuracy: {performance_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {performance_metrics['f1_score']:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(performance_metrics, f, default_flow_style=False)
    
    return performance_metrics


def generate_explanations(
    model: InterpretableBERT,
    test_dataset: NLPDataset,
    config: Dict[str, Any],
    output_dir: str,
    n_samples: int = 10,
) -> Dict[str, Any]:
    """Generate explanations for test samples.
    
    Args:
        model: The trained model.
        test_dataset: Test dataset.
        config: Explanation configuration.
        output_dir: Output directory for explanations.
        
    Returns:
        Dictionary containing explanations.
    """
    print("Generating explanations...")
    
    # Create explainer
    explainer = NLPExplainer(model)
    
    # Select samples for explanation
    sample_texts = test_dataset.texts[:n_samples]
    sample_labels = test_dataset.labels[:n_samples]
    
    # Generate explanations
    explanations = {}
    
    # SHAP explanations
    if config.get("use_shap", True):
        print("Generating SHAP explanations...")
        shap_explanations = explainer.explain_with_shap(
            sample_texts,
            method="kernel",
            max_samples=config.get("shap_max_samples", 100),
        )
        explanations["shap"] = shap_explanations
    
    # LIME explanations
    if config.get("use_lime", True):
        print("Generating LIME explanations...")
        lime_explanations = explainer.explain_with_lime(
            sample_texts,
            num_features=config.get("lime_num_features", 10),
            class_names=test_dataset.label_names,
        )
        explanations["lime"] = lime_explanations
    
    # Attention explanations
    if config.get("use_attention", True):
        print("Generating attention explanations...")
        attention_explanations = explainer.explain_with_attention(
            sample_texts,
            layer=config.get("attention_layer", -1),
        )
        explanations["attention"] = attention_explanations
    
    # Integrated Gradients
    if config.get("use_ig", True):
        print("Generating Integrated Gradients explanations...")
        ig_explanations = explainer.explain_with_integrated_gradients(
            sample_texts,
            n_steps=config.get("ig_n_steps", 50),
        )
        explanations["ig"] = ig_explanations
    
    # Save explanations
    explanations_path = os.path.join(output_dir, "explanations.yaml")
    with open(explanations_path, "w") as f:
        # Convert non-serializable objects to strings
        serializable_explanations = {}
        for method, exp_data in explanations.items():
            serializable_explanations[method] = {
                "texts": exp_data.get("texts", []),
                "method": exp_data.get("method", method),
            }
        yaml.dump(serializable_explanations, f, default_flow_style=False)
    
    return explanations


def create_visualizations(
    model: InterpretableBERT,
    test_dataset: NLPDataset,
    explanations: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    n_samples: int = 5,
) -> None:
    """Create visualizations for explanations.
    
    Args:
        model: The trained model.
        test_dataset: Test dataset.
        explanations: Dictionary containing explanations.
        config: Visualization configuration.
        output_dir: Output directory for visualizations.
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    print("Creating visualizations...")
    
    # Create visualizer
    visualizer = NLPVisualizer(model.tokenizer)
    
    # Create assets directory
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Generate visualizations for sample texts
    sample_texts = test_dataset.texts[:n_samples]
    
    for i, text in enumerate(sample_texts):
        print(f"Creating visualizations for sample {i+1}/{n_samples}")
        
        # Create explanation comparison plot
        comparison_plot = visualizer.plot_explanation_comparison(
            text,
            explanations,
            methods=["shap", "lime", "attention"],
        )
        
        comparison_path = os.path.join(assets_dir, f"explanation_comparison_{i}.png")
        comparison_plot.savefig(comparison_path, bbox_inches="tight", dpi=150)
        plt.close(comparison_plot)
        
        # Create individual method plots
        if "shap" in explanations and "shap_values" in explanations["shap"]:
            shap_plot = visualizer.plot_shap_values(
                text,
                explanations["shap"]["shap_values"][i] if i < len(explanations["shap"]["shap_values"]) else explanations["shap"]["shap_values"][0],
            )
            shap_path = os.path.join(assets_dir, f"shap_explanation_{i}.png")
            shap_plot.savefig(shap_path, bbox_inches="tight", dpi=150)
            plt.close(shap_plot)
        
        if "lime" in explanations and "explanations" in explanations["lime"]:
            lime_plot = visualizer.plot_lime_explanation(
                explanations["lime"]["explanations"][i] if i < len(explanations["lime"]["explanations"]) else explanations["lime"]["explanations"][0],
            )
            lime_path = os.path.join(assets_dir, f"lime_explanation_{i}.png")
            lime_plot.savefig(lime_path, bbox_inches="tight", dpi=150)
            plt.close(lime_plot)
        
        if "attention" in explanations and "explanations" in explanations["attention"]:
            attention_data = explanations["attention"]["explanations"][i] if i < len(explanations["attention"]["explanations"]) else explanations["attention"]["explanations"][0]
            attention_plot = visualizer.plot_attention_heatmap(
                text,
                attention_data["attention_weights"],
                attention_data["tokens"],
            )
            attention_path = os.path.join(assets_dir, f"attention_explanation_{i}.png")
            attention_plot.savefig(attention_path, bbox_inches="tight", dpi=150)
            plt.close(attention_plot)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train Interpretable NLP Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="bert_experiment", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic dataset")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "model_name": "bert-base-uncased",
                "num_labels": 20,
                "dropout_rate": 0.1,
                "freeze_bert": False,
            },
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "max_length": 512,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "logging_steps": 100,
                "eval_steps": 500,
                "save_steps": 1000,
            },
            "explanation": {
                "use_shap": True,
                "use_lime": True,
                "use_attention": True,
                "use_ig": True,
                "shap_max_samples": 100,
                "lime_num_features": 10,
                "attention_layer": -1,
                "ig_n_steps": 50,
            },
            "evaluation": {
                "n_samples": 10,
                "n_visualization_samples": 5,
            },
        }
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.experiment_name)
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    if args.use_synthetic:
        print("Loading synthetic dataset...")
        dataset = create_synthetic_dataset(
            n_samples=1000,
            n_classes=5,
            random_state=args.seed,
        )
    else:
        print("Loading 20 Newsgroups dataset...")
        dataset = load_20newsgroups()
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        test_size=0.2,
        val_size=0.1,
        random_state=args.seed,
    )
    
    print(f"Training samples: {len(train_dataset.texts)}")
    print(f"Validation samples: {len(val_dataset.texts)}")
    print(f"Test samples: {len(test_dataset.texts)}")
    
    # Create model
    print("Creating model...")
    model = InterpretableBERT(
        model_name=config["model"]["model_name"],
        num_labels=config["model"]["num_labels"],
        dropout_rate=config["model"]["dropout_rate"],
        freeze_bert=config["model"]["freeze_bert"],
    )
    
    # Train model
    trained_model = train_model(
        model,
        train_dataset,
        val_dataset,
        config["training"],
        output_dir,
    )
    
    # Evaluate model
    performance_metrics = evaluate_model(
        trained_model,
        test_dataset,
        config["evaluation"],
        output_dir,
    )
    
    # Generate explanations
    explanations = generate_explanations(
        trained_model,
        test_dataset,
        config["explanation"],
        output_dir,
        n_samples=config["evaluation"]["n_samples"],
    )
    
    # Create visualizations
    create_visualizations(
        trained_model,
        test_dataset,
        explanations,
        config["explanation"],
        output_dir,
        n_samples=config["evaluation"]["n_visualization_samples"],
    )
    
    print("Training pipeline completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
