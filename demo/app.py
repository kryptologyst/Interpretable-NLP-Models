"""Streamlit demo application for Interpretable NLP Models."""

import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoTokenizer

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import NLPDataset, load_20newsgroups, create_synthetic_dataset
from src.models.bert import InterpretableBERT
from src.explainers.nlp_explainer import NLPExplainer
from src.eval.metrics import ModelEvaluator
from src.viz.visualizer import NLPVisualizer
from src.utils.device import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Interpretable NLP Models",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This is a research and educational tool for Explainable AI (XAI). The explanations provided may be:
- Unstable or inconsistent across runs
- Misleading or incomplete
- Not suitable for high-stakes decisions

**DO NOT** use these explanations for regulated decisions without human review and validation.
"""


@st.cache_resource
def load_model_and_data(model_path: str, use_synthetic: bool = False) -> Tuple[InterpretableBERT, NLPDataset]:
    """Load model and data with caching."""
    try:
        if os.path.exists(model_path):
            model = InterpretableBERT.load_model(model_path)
        else:
            # Create a new model for demo
            model = InterpretableBERT(
                model_name="bert-base-uncased",
                num_labels=20 if not use_synthetic else 5,
            )
        
        if use_synthetic:
            dataset = create_synthetic_dataset(n_samples=200, n_classes=5)
        else:
            dataset = load_20newsgroups()
        
        return model, dataset
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Interpretable NLP Models</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Model Type",
        ["Pre-trained BERT", "Custom Model"],
        help="Choose between pre-trained BERT or load a custom trained model"
    )
    
    use_synthetic = st.sidebar.checkbox(
        "Use Synthetic Dataset",
        value=False,
        help="Use synthetic dataset for faster demo (5 classes instead of 20)"
    )
    
    # Load model and data
    if model_option == "Pre-trained BERT":
        model_path = ""
    else:
        model_path = st.sidebar.text_input(
            "Model Path",
            value="outputs/bert_experiment/final_model",
            help="Path to trained model directory"
        )
    
    model, dataset = load_model_and_data(model_path, use_synthetic)
    
    if model is None or dataset is None:
        st.error("Failed to load model or dataset. Please check the configuration.")
        return
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )
    
    # Create explainer
    explainer = NLPExplainer(model)
    visualizer = NLPVisualizer(model.tokenizer)
    evaluator = ModelEvaluator()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance", 
        "🔍 Local Explanations", 
        "🌍 Global Explanations", 
        "📈 Evaluation Metrics",
        "⚙️ Model Training"
    ])
    
    with tab1:
        st.header("Model Performance Overview")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            predictions = model.predict(test_dataset.texts)
            accuracy = np.mean(predictions == test_dataset.labels)
            st.metric("Test Accuracy", f"{accuracy:.3f}")
        
        with col2:
            from sklearn.metrics import f1_score
            f1 = f1_score(test_dataset.labels, predictions, average="weighted")
            st.metric("F1 Score", f"{f1:.3f}")
        
        with col3:
            st.metric("Dataset Size", len(dataset.texts))
        
        with col4:
            st.metric("Number of Classes", dataset.n_classes)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_dataset.labels, predictions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Class distribution
        st.subheader("Class Distribution")
        class_counts = np.bincount(test_dataset.labels)
        class_names = test_dataset.label_names
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(class_names)), class_counts)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Test Set Class Distribution")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha="center", va="bottom")
        
        st.pyplot(fig)
    
    with tab2:
        st.header("Local Explanations")
        
        # Text input
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text to explain:",
            value="The new software update includes several bug fixes and performance improvements.",
            height=100,
        )
        
        if st.button("Generate Explanations"):
            if input_text.strip():
                with st.spinner("Generating explanations..."):
                    # Get prediction
                    prediction, probabilities = model.predict([input_text], return_probabilities=True)
                    predicted_class = prediction[0]
                    confidence = probabilities[0][predicted_class]
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", dataset.label_names[predicted_class])
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    # Explanation methods
                    methods = st.multiselect(
                        "Explanation Methods",
                        ["SHAP", "LIME", "Attention", "Integrated Gradients"],
                        default=["SHAP", "LIME", "Attention"],
                    )
                    
                    if methods:
                        explanations = {}
                        
                        # Generate explanations
                        if "SHAP" in methods:
                            try:
                                shap_exp = explainer.explain_with_shap([input_text], method="kernel", max_samples=50)
                                explanations["shap"] = shap_exp
                            except Exception as e:
                                st.error(f"SHAP explanation failed: {e}")
                        
                        if "LIME" in methods:
                            try:
                                lime_exp = explainer.explain_with_lime([input_text], class_names=dataset.label_names)
                                explanations["lime"] = lime_exp
                            except Exception as e:
                                st.error(f"LIME explanation failed: {e}")
                        
                        if "Attention" in methods:
                            try:
                                att_exp = explainer.explain_with_attention([input_text])
                                explanations["attention"] = att_exp
                            except Exception as e:
                                st.error(f"Attention explanation failed: {e}")
                        
                        if "Integrated Gradients" in methods:
                            try:
                                ig_exp = explainer.explain_with_integrated_gradients([input_text])
                                explanations["ig"] = ig_exp
                            except Exception as e:
                                st.error(f"Integrated Gradients explanation failed: {e}")
                        
                        # Display explanations
                        if explanations:
                            # Create comparison plot
                            comparison_fig = visualizer.plot_explanation_comparison(
                                input_text,
                                explanations,
                                methods=[m.lower() for m in methods],
                            )
                            st.pyplot(comparison_fig)
                            
                            # Individual explanations
                            for method in methods:
                                st.subheader(f"{method} Explanation")
                                
                                if method.lower() == "shap" and "shap" in explanations:
                                    if "shap_values" in explanations["shap"]:
                                        shap_fig = visualizer.plot_shap_values(
                                            input_text,
                                            explanations["shap"]["shap_values"][0],
                                        )
                                        st.pyplot(shap_fig)
                                
                                elif method.lower() == "lime" and "lime" in explanations:
                                    if "explanations" in explanations["lime"] and explanations["lime"]["explanations"][0]:
                                        lime_fig = visualizer.plot_lime_explanation(
                                            explanations["lime"]["explanations"][0],
                                        )
                                        st.pyplot(lime_fig)
                                
                                elif method.lower() == "attention" and "attention" in explanations:
                                    if "explanations" in explanations["attention"]:
                                        att_data = explanations["attention"]["explanations"][0]
                                        att_fig = visualizer.plot_attention_heatmap(
                                            input_text,
                                            att_data["attention_weights"],
                                            att_data["tokens"],
                                        )
                                        st.pyplot(att_fig)
                                
                                elif method.lower() == "integrated gradients" and "ig" in explanations:
                                    if "explanations" in explanations["ig"] and explanations["ig"]["explanations"][0]:
                                        ig_data = explanations["ig"]["explanations"][0]
                                        ig_fig = visualizer.plot_integrated_gradients(
                                            input_text,
                                            ig_data["attributions"],
                                            ig_data["tokens"],
                                        )
                                        st.pyplot(ig_fig)
            else:
                st.warning("Please enter some text to explain.")
    
    with tab3:
        st.header("Global Explanations")
        
        # Sample selection
        st.subheader("Sample Selection")
        n_samples = st.slider("Number of samples to analyze", 5, 50, 20)
        
        # Random sample selection
        sample_indices = np.random.choice(len(test_dataset.texts), n_samples, replace=False)
        sample_texts = [test_dataset.texts[i] for i in sample_indices]
        sample_labels = [test_dataset.labels[i] for i in sample_indices]
        
        # Display samples
        st.subheader("Selected Samples")
        for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
            with st.expander(f"Sample {i+1}: {dataset.label_names[label]}"):
                st.text(text[:200] + "..." if len(text) > 200 else text)
        
        # Global explanation methods
        global_methods = st.multiselect(
            "Global Explanation Methods",
            ["SHAP Summary", "Attention Analysis", "Feature Importance"],
            default=["SHAP Summary"],
        )
        
        if st.button("Generate Global Explanations"):
            with st.spinner("Generating global explanations..."):
                
                if "SHAP Summary" in global_methods:
                    st.subheader("SHAP Summary")
                    try:
                        shap_explanations = explainer.explain_with_shap(
                            sample_texts,
                            method="kernel",
                            max_samples=50,
                        )
                        
                        if "shap_values" in shap_explanations:
                            # Create summary plot
                            shap_values = shap_explanations["shap_values"]
                            
                            # Average SHAP values across samples
                            avg_shap_values = np.mean(np.abs(shap_values), axis=0)
                            
                            # Get tokens (simplified - using first sample)
                            tokens = model.tokenizer.tokenize(sample_texts[0])
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            bars = ax.bar(range(len(tokens)), avg_shap_values[:len(tokens)])
                            ax.set_xlabel("Token Position")
                            ax.set_ylabel("Average |SHAP Value|")
                            ax.set_title("Average SHAP Values Across Samples")
                            ax.set_xticks(range(len(tokens)))
                            ax.set_xticklabels(tokens, rotation=45, ha="right")
                            
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"SHAP summary failed: {e}")
                
                if "Attention Analysis" in global_methods:
                    st.subheader("Attention Analysis")
                    try:
                        attention_explanations = explainer.explain_with_attention(sample_texts)
                        
                        if "explanations" in attention_explanations:
                            # Average attention weights
                            all_attentions = []
                            for exp in attention_explanations["explanations"]:
                                if exp and "attention_weights" in exp:
                                    all_attentions.append(exp["attention_weights"])
                            
                            if all_attentions:
                                avg_attention = np.mean(all_attentions, axis=0)
                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                im = ax.imshow(avg_attention.reshape(1, -1), cmap="Blues", aspect="auto")
                                ax.set_title("Average Attention Weights")
                                ax.set_xlabel("Token Position")
                                ax.set_ylabel("Attention")
                                
                                # Add colorbar
                                plt.colorbar(im, ax=ax)
                                st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Attention analysis failed: {e}")
    
    with tab4:
        st.header("Evaluation Metrics")
        
        # Calculate comprehensive metrics
        with st.spinner("Calculating evaluation metrics..."):
            performance_metrics = evaluator.evaluate_model(
                model,
                test_dataset.texts,
                test_dataset.labels,
                test_dataset.label_names,
            )
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Metrics")
            st.metric("Accuracy", f"{performance_metrics['accuracy']:.3f}")
            st.metric("Precision", f"{performance_metrics['precision']:.3f}")
            st.metric("Recall", f"{performance_metrics['recall']:.3f}")
            st.metric("F1 Score", f"{performance_metrics['f1_score']:.3f}")
        
        with col2:
            st.subheader("Calibration Metrics")
            if "calibration_metrics" in performance_metrics:
                calib = performance_metrics["calibration_metrics"]
                st.metric("Expected Calibration Error", f"{calib.get('ece', 0):.3f}")
                st.metric("Maximum Calibration Error", f"{calib.get('mce', 0):.3f}")
                st.metric("Brier Score", f"{calib.get('brier_score', 0):.3f}")
        
        # Robustness metrics
        st.subheader("Robustness Metrics")
        if "robustness_metrics" in performance_metrics:
            robust = performance_metrics["robustness_metrics"]
            for key, value in robust.items():
                if "accuracy_drop" in key:
                    st.metric(key.replace("_", " ").title(), f"{value:.3f}")
        
        # Detailed classification report
        st.subheader("Detailed Classification Report")
        if "classification_report" in performance_metrics:
            report = performance_metrics["classification_report"]
            
            # Convert to DataFrame for better display
            import pandas as pd
            
            # Extract per-class metrics
            class_metrics = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and "precision" in metrics:
                    class_metrics.append({
                        "Class": class_name,
                        "Precision": f"{metrics['precision']:.3f}",
                        "Recall": f"{metrics['recall']:.3f}",
                        "F1-Score": f"{metrics['f1-score']:.3f}",
                        "Support": int(metrics['support']),
                    })
            
            if class_metrics:
                df = pd.DataFrame(class_metrics)
                st.dataframe(df, use_container_width=True)
    
    with tab5:
        st.header("Model Training")
        
        st.info("This is a demo application. For actual model training, use the training script:")
        st.code("python scripts/train.py --config configs/default.yaml")
        
        # Training configuration
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
            learning_rate = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5], index=1)
        
        with col2:
            max_length = st.number_input("Max Sequence Length", min_value=128, max_value=512, value=512)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
            warmup_steps = st.number_input("Warmup Steps", min_value=100, max_value=1000, value=500)
        
        # Model architecture
        st.subheader("Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "Model Name",
                ["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased"],
                index=0,
            )
            dropout_rate = st.number_input("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        with col2:
            freeze_bert = st.checkbox("Freeze BERT Parameters", value=False)
            num_labels = st.number_input("Number of Labels", min_value=2, max_value=100, value=20)
        
        # Training options
        st.subheader("Training Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_synthetic_data = st.checkbox("Use Synthetic Dataset", value=False)
            save_model = st.checkbox("Save Model", value=True)
        
        with col2:
            generate_explanations = st.checkbox("Generate Explanations", value=True)
            create_visualizations = st.checkbox("Create Visualizations", value=True)
        
        # Training button (demo only)
        if st.button("Start Training (Demo)", disabled=True):
            st.info("This is a demo. Use the training script for actual training.")
        
        # Display configuration
        st.subheader("Current Configuration")
        config = {
            "model": {
                "model_name": model_name,
                "num_labels": num_labels,
                "dropout_rate": dropout_rate,
                "freeze_bert": freeze_bert,
            },
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
            },
            "data": {
                "use_synthetic": use_synthetic_data,
            },
            "output": {
                "save_model": save_model,
                "generate_explanations": generate_explanations,
                "create_visualizations": create_visualizations,
            },
        }
        
        st.json(config)


if __name__ == "__main__":
    main()
