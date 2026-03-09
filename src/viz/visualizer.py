"""Visualization functions for NLP model explanations."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

# Set style
plt.style.use("default")
sns.set_palette("husl")


class NLPVisualizer:
    """Comprehensive visualizer for NLP model explanations."""
    
    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
    ):
        """Initialize visualizer.
        
        Args:
            tokenizer: Tokenizer for text processing.
            figsize: Default figure size.
            dpi: Figure DPI.
        """
        self.tokenizer = tokenizer
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_attention_heatmap(
        self,
        text: str,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
        layer: int = -1,
        head: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot attention heatmap for a text.
        
        Args:
            text: Input text.
            attention_weights: Attention weight matrix.
            tokens: List of tokens (if None, will tokenize text).
            layer: BERT layer number.
            head: Attention head number (None for average).
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if tokens is None:
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
            else:
                tokens = text.split()
        
        # Ensure attention weights match token length
        if len(attention_weights) != len(tokens):
            attention_weights = attention_weights[:len(tokens)]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(
            attention_weights.reshape(1, -1),
            cmap="Blues",
            aspect="auto",
            interpolation="nearest",
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels(["Attention"])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)
        
        # Add title
        title = f"Attention Heatmap - Layer {layer}"
        if head is not None:
            title += f", Head {head}"
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def plot_shap_values(
        self,
        text: str,
        shap_values: np.ndarray,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot SHAP values for a text.
        
        Args:
            text: Input text.
            shap_values: SHAP values array.
            tokens: List of tokens (if None, will tokenize text).
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if tokens is None:
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text)
            else:
                tokens = text.split()
        
        # Ensure shap values match token length
        if len(shap_values) != len(tokens):
            shap_values = shap_values[:len(tokens)]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create bar plot
        colors = ["red" if val < 0 else "blue" for val in shap_values]
        bars = ax.bar(range(len(tokens)), shap_values, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("SHAP Value")
        ax.set_title("SHAP Values for Text Classification")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.01 if height >= 0 else -0.01),
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def plot_lime_explanation(
        self,
        lime_explanation,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot LIME explanation.
        
        Args:
            lime_explanation: LIME explanation object.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if lime_explanation is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, "No LIME explanation available", ha="center", va="center")
            ax.set_title("LIME Explanation")
            return fig
        
        # Extract explanation data
        explanation_list = lime_explanation.as_list()
        features = [item[0] for item in explanation_list]
        scores = [item[1] for item in explanation_list]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create horizontal bar plot
        colors = ["red" if score < 0 else "blue" for score in scores]
        bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("LIME Score")
        ax.set_title("LIME Explanation")
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(
                width + (0.01 if width >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left" if width >= 0 else "right",
                va="center",
                fontsize=8,
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def plot_integrated_gradients(
        self,
        text: str,
        attributions: np.ndarray,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Integrated Gradients attributions.
        
        Args:
            text: Input text.
            attributions: Attribution values.
            tokens: List of tokens (if None, will tokenize text).
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if tokens is None:
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text)
            else:
                tokens = text.split()
        
        # Ensure attributions match token length
        if len(attributions) != len(tokens):
            attributions = attributions[:len(tokens)]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create bar plot
        colors = ["red" if val < 0 else "blue" for val in attributions]
        bars = ax.bar(range(len(tokens)), attributions, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Attribution Value")
        ax.set_title("Integrated Gradients Attribution")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, attributions)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.01 if height >= 0 else -0.01),
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def plot_model_performance(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot model performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # Accuracy metrics
        accuracy_metrics = ["accuracy", "precision", "recall", "f1_score"]
        values = [metrics.get(metric, 0) for metric in accuracy_metrics]
        
        axes[0, 0].bar(accuracy_metrics, values, color="skyblue", alpha=0.7)
        axes[0, 0].set_title("Classification Metrics")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        
        # Confusion Matrix
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1])
            axes[0, 1].set_title("Confusion Matrix")
            axes[0, 1].set_xlabel("Predicted")
            axes[0, 1].set_ylabel("Actual")
        
        # Calibration metrics
        if "calibration_metrics" in metrics:
            calib_metrics = metrics["calibration_metrics"]
            calib_names = list(calib_metrics.keys())
            calib_values = list(calib_metrics.values())
            
            axes[1, 0].bar(calib_names, calib_values, color="lightcoral", alpha=0.7)
            axes[1, 0].set_title("Calibration Metrics")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].tick_params(axis="x", rotation=45)
        
        # Robustness metrics
        if "robustness_metrics" in metrics:
            robust_metrics = metrics["robustness_metrics"]
            robust_names = [k for k in robust_metrics.keys() if "accuracy_drop" in k]
            robust_values = [robust_metrics[k] for k in robust_names]
            
            axes[1, 1].bar(robust_names, robust_values, color="lightgreen", alpha=0.7)
            axes[1, 1].set_title("Robustness Metrics (Accuracy Drop)")
            axes[1, 1].set_ylabel("Accuracy Drop")
            axes[1, 1].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def plot_explanation_comparison(
        self,
        text: str,
        explanations: Dict[str, Any],
        methods: List[str] = ["shap", "lime", "attention"],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comparison of different explanation methods.
        
        Args:
            text: Input text.
            explanations: Dictionary containing explanations from different methods.
            methods: List of methods to compare.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        n_methods = len(methods)
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 4 * n_methods), dpi=self.dpi)
        
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            if method not in explanations:
                axes[i].text(0.5, 0.5, f"No {method.upper()} explanation available", 
                           ha="center", va="center")
                axes[i].set_title(f"{method.upper()} Explanation")
                continue
            
            if method == "shap" and "shap_values" in explanations[method]:
                self._plot_shap_subplot(axes[i], text, explanations[method]["shap_values"])
            elif method == "lime" and "explanations" in explanations[method]:
                self._plot_lime_subplot(axes[i], explanations[method]["explanations"][0])
            elif method == "attention" and "explanations" in explanations[method]:
                self._plot_attention_subplot(axes[i], text, explanations[method]["explanations"][0])
            else:
                axes[i].text(0.5, 0.5, f"Incomplete {method.upper()} explanation", 
                           ha="center", va="center")
                axes[i].set_title(f"{method.upper()} Explanation")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
        
        return fig
    
    def _plot_shap_subplot(self, ax, text: str, shap_values: np.ndarray):
        """Plot SHAP values in subplot."""
        tokens = self.tokenizer.tokenize(text) if self.tokenizer else text.split()
        
        if len(shap_values) != len(tokens):
            shap_values = shap_values[:len(tokens)]
        
        colors = ["red" if val < 0 else "blue" for val in shap_values]
        bars = ax.bar(range(len(tokens)), shap_values, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("SHAP Value")
        ax.set_title("SHAP Explanation")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    
    def _plot_lime_subplot(self, ax, lime_explanation):
        """Plot LIME explanation in subplot."""
        if lime_explanation is None:
            ax.text(0.5, 0.5, "No LIME explanation", ha="center", va="center")
            ax.set_title("LIME Explanation")
            return
        
        explanation_list = lime_explanation.as_list()
        features = [item[0] for item in explanation_list]
        scores = [item[1] for item in explanation_list]
        
        colors = ["red" if score < 0 else "blue" for score in scores]
        bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("LIME Score")
        ax.set_title("LIME Explanation")
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    
    def _plot_attention_subplot(self, ax, text: str, attention_data: Dict[str, Any]):
        """Plot attention weights in subplot."""
        tokens = attention_data.get("tokens", text.split())
        attention_weights = attention_data.get("attention_weights", [])
        
        if len(attention_weights) != len(tokens):
            attention_weights = attention_weights[:len(tokens)]
        
        im = ax.imshow(
            np.array(attention_weights).reshape(1, -1),
            cmap="Blues",
            aspect="auto",
            interpolation="nearest",
        )
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels(["Attention"])
        ax.set_title("Attention Explanation")
        
        plt.colorbar(im, ax=ax)
    
    def create_interactive_plot(
        self,
        text: str,
        explanations: Dict[str, Any],
        methods: List[str] = ["shap", "lime", "attention"],
    ) -> go.Figure:
        """Create interactive Plotly visualization.
        
        Args:
            text: Input text.
            explanations: Dictionary containing explanations.
            methods: List of methods to visualize.
            
        Returns:
            Plotly Figure object.
        """
        n_methods = len(methods)
        fig = make_subplots(
            rows=n_methods,
            cols=1,
            subplot_titles=[f"{method.upper()} Explanation" for method in methods],
            vertical_spacing=0.1,
        )
        
        for i, method in enumerate(methods):
            if method not in explanations:
                continue
            
            if method == "shap" and "shap_values" in explanations[method]:
                tokens = self.tokenizer.tokenize(text) if self.tokenizer else text.split()
                shap_values = explanations[method]["shap_values"]
                
                if len(shap_values) != len(tokens):
                    shap_values = shap_values[:len(tokens)]
                
                colors = ["red" if val < 0 else "blue" for val in shap_values]
                
                fig.add_trace(
                    go.Bar(
                        x=tokens,
                        y=shap_values,
                        marker_color=colors,
                        name=f"{method}_values",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )
        
        fig.update_layout(
            height=300 * n_methods,
            title_text="Interactive Explanation Comparison",
            showlegend=False,
        )
        
        return fig
