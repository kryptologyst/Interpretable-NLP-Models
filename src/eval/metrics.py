"""Evaluation metrics for interpretability methods."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
import torchmetrics

from ..utils.device import get_device

warnings.filterwarnings("ignore", category=UserWarning)


class InterpretabilityMetrics:
    """Comprehensive metrics for evaluating interpretability methods."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize metrics calculator.
        
        Args:
            device: Device for computations.
        """
        self.device = device or get_device()
    
    def faithfulness_metrics(
        self,
        model,
        texts: List[str],
        explanations: Dict[str, Any],
        method: str = "deletion",
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Calculate faithfulness metrics for explanations.
        
        Args:
            model: The model to evaluate.
            texts: List of input texts.
            explanations: Dictionary containing explanations.
            method: Faithfulness method ('deletion' or 'insertion').
            top_k: Number of top features to consider.
            
        Returns:
            Dictionary containing faithfulness metrics.
        """
        if method not in ["deletion", "insertion"]:
            raise ValueError("Method must be 'deletion' or 'insertion'")
        
        original_predictions, _ = model.predict(texts, return_probabilities=True)
        original_confidences = [pred[original_predictions[i]] for i, pred in enumerate(original_predictions)]
        
        modified_confidences = []
        
        for i, text in enumerate(texts):
            if method == "deletion":
                # Remove top-k important features
                modified_text = self._remove_top_features(text, explanations, i, top_k)
            else:
                # Keep only top-k important features
                modified_text = self._keep_top_features(text, explanations, i, top_k)
            
            modified_pred, modified_prob = model.predict([modified_text], return_probabilities=True)
            modified_confidences.append(modified_prob[0][original_predictions[i]])
        
        # Calculate faithfulness metrics
        confidence_drops = [orig - mod for orig, mod in zip(original_confidences, modified_confidences)]
        
        return {
            "mean_confidence_drop": np.mean(confidence_drops),
            "std_confidence_drop": np.std(confidence_drops),
            "faithfulness_score": np.mean(confidence_drops),  # Higher is better for deletion
        }
    
    def stability_metrics(
        self,
        explainer,
        texts: List[str],
        n_runs: int = 5,
        method: str = "shap",
    ) -> Dict[str, float]:
        """Calculate stability metrics across multiple runs.
        
        Args:
            explainer: The explainer to evaluate.
            texts: List of input texts.
            n_runs: Number of runs for stability test.
            method: Explanation method to test.
            
        Returns:
            Dictionary containing stability metrics.
        """
        all_explanations = []
        
        for run in range(n_runs):
            if method == "shap":
                explanations = explainer.explain_with_shap(texts)
            elif method == "lime":
                explanations = explainer.explain_with_lime(texts)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            all_explanations.append(explanations)
        
        # Calculate stability metrics
        stability_scores = []
        
        for i in range(len(texts)):
            explanations_i = [run["explanations"][i] for run in all_explanations if "explanations" in run]
            
            if len(explanations_i) > 1:
                # Calculate correlation between explanations
                correlations = []
                for j in range(len(explanations_i)):
                    for k in range(j + 1, len(explanations_i)):
                        if method == "shap":
                            corr = np.corrcoef(explanations_i[j], explanations_i[k])[0, 1]
                        else:
                            # For LIME, use feature importance scores
                            corr = self._correlate_lime_explanations(explanations_i[j], explanations_i[k])
                        
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                stability_scores.append(np.mean(correlations) if correlations else 0.0)
        
        return {
            "mean_stability": np.mean(stability_scores),
            "std_stability": np.std(stability_scores),
            "min_stability": np.min(stability_scores),
        }
    
    def fidelity_metrics(
        self,
        original_model,
        surrogate_model,
        texts: List[str],
        labels: List[int],
    ) -> Dict[str, float]:
        """Calculate fidelity metrics between original and surrogate models.
        
        Args:
            original_model: The original model.
            surrogate_model: The surrogate model.
            texts: List of input texts.
            labels: Ground truth labels.
            
        Returns:
            Dictionary containing fidelity metrics.
        """
        # Get predictions from both models
        original_preds = original_model.predict(texts)
        surrogate_preds = surrogate_model.predict(texts)
        
        # Calculate agreement metrics
        agreement = np.mean(original_preds == surrogate_preds)
        
        # Calculate accuracy metrics
        original_accuracy = accuracy_score(labels, original_preds)
        surrogate_accuracy = accuracy_score(labels, surrogate_preds)
        
        # Calculate prediction correlation
        pred_correlation = np.corrcoef(original_preds, surrogate_preds)[0, 1]
        
        return {
            "agreement": agreement,
            "original_accuracy": original_accuracy,
            "surrogate_accuracy": surrogate_accuracy,
            "accuracy_drop": original_accuracy - surrogate_accuracy,
            "prediction_correlation": pred_correlation,
        }
    
    def calibration_metrics(
        self,
        model,
        texts: List[str],
        labels: List[int],
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Calculate calibration metrics for model predictions.
        
        Args:
            model: The model to evaluate.
            texts: List of input texts.
            labels: Ground truth labels.
            n_bins: Number of bins for calibration.
            
        Returns:
            Dictionary containing calibration metrics.
        """
        predictions, probabilities = model.predict(texts, return_probabilities=True)
        
        # Get maximum probabilities (confidence scores)
        confidences = np.max(probabilities, axis=1)
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, predictions, labels, n_bins)
        
        # Calculate Maximum Calibration Error (MCE)
        mce = self._calculate_mce(confidences, predictions, labels, n_bins)
        
        # Calculate Brier Score
        brier_score = self._calculate_brier_score(probabilities, labels)
        
        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier_score,
        }
    
    def robustness_metrics(
        self,
        model,
        texts: List[str],
        labels: List[int],
        noise_levels: List[float] = [0.01, 0.05, 0.1],
    ) -> Dict[str, Any]:
        """Calculate robustness metrics for model predictions.
        
        Args:
            model: The model to evaluate.
            texts: List of input texts.
            labels: Ground truth labels.
            noise_levels: List of noise levels to test.
            
        Returns:
            Dictionary containing robustness metrics.
        """
        original_predictions = model.predict(texts)
        original_accuracy = accuracy_score(labels, original_predictions)
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            # Add noise to texts (simplified - in practice, use proper adversarial methods)
            noisy_texts = self._add_noise_to_texts(texts, noise_level)
            noisy_predictions = model.predict(noisy_texts)
            noisy_accuracy = accuracy_score(labels, noisy_predictions)
            
            robustness_results[f"accuracy_at_noise_{noise_level}"] = noisy_accuracy
            robustness_results[f"accuracy_drop_at_noise_{noise_level}"] = original_accuracy - noisy_accuracy
        
        robustness_results["original_accuracy"] = original_accuracy
        
        return robustness_results
    
    def _remove_top_features(
        self,
        text: str,
        explanations: Dict[str, Any],
        text_idx: int,
        top_k: int,
    ) -> str:
        """Remove top-k important features from text."""
        # This is a simplified implementation
        # In practice, you would need to implement proper feature removal
        words = text.split()
        if len(words) <= top_k:
            return " ".join(words[:1])  # Keep at least one word
        return " ".join(words[top_k:])
    
    def _keep_top_features(
        self,
        text: str,
        explanations: Dict[str, Any],
        text_idx: int,
        top_k: int,
    ) -> str:
        """Keep only top-k important features from text."""
        # This is a simplified implementation
        words = text.split()
        if len(words) <= top_k:
            return text
        return " ".join(words[:top_k])
    
    def _correlate_lime_explanations(self, exp1, exp2) -> float:
        """Calculate correlation between two LIME explanations."""
        if exp1 is None or exp2 is None:
            return 0.0
        
        # Extract feature importance scores from LIME explanations
        features1 = exp1.as_list()
        features2 = exp2.as_list()
        
        # Create feature score vectors
        all_features = set([f[0] for f in features1] + [f[0] for f in features2])
        scores1 = {f[0]: f[1] for f in features1}
        scores2 = {f[0]: f[1] for f in features2}
        
        vector1 = [scores1.get(f, 0) for f in all_features]
        vector2 = [scores2.get(f, 0) for f in all_features]
        
        return np.corrcoef(vector1, vector2)[0, 1]
    
    def _calculate_ece(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int,
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int,
    ) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_brier_score(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Calculate Brier Score."""
        n_classes = probabilities.shape[1]
        labels_one_hot = np.eye(n_classes)[labels]
        
        return np.mean(np.sum((probabilities - labels_one_hot) ** 2, axis=1))
    
    def _add_noise_to_texts(self, texts: List[str], noise_level: float) -> List[str]:
        """Add noise to texts (simplified implementation)."""
        noisy_texts = []
        for text in texts:
            words = text.split()
            n_words_to_change = max(1, int(len(words) * noise_level))
            
            # Randomly replace some words
            indices_to_change = np.random.choice(
                len(words), 
                min(n_words_to_change, len(words)), 
                replace=False
            )
            
            noisy_words = words.copy()
            for idx in indices_to_change:
                noisy_words[idx] = f"noise_{np.random.randint(1000)}"
            
            noisy_texts.append(" ".join(noisy_words))
        
        return noisy_texts


class ModelEvaluator:
    """Comprehensive model evaluation with interpretability metrics."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            device: Device for computations.
        """
        self.device = device or get_device()
        self.metrics = InterpretabilityMetrics(device)
    
    def evaluate_model(
        self,
        model,
        texts: List[str],
        labels: List[int],
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: The model to evaluate.
            texts: List of input texts.
            labels: Ground truth labels.
            label_names: Names of labels.
            
        Returns:
            Dictionary containing all evaluation metrics.
        """
        predictions = model.predict(texts)
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, 
            predictions, 
            target_names=label_names,
            output_dict=True
        )
        
        # Calibration metrics
        calibration_metrics = self.metrics.calibration_metrics(model, texts, labels)
        
        # Robustness metrics
        robustness_metrics = self.metrics.robustness_metrics(model, texts, labels)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "calibration_metrics": calibration_metrics,
            "robustness_metrics": robustness_metrics,
        }
    
    def evaluate_explanations(
        self,
        model,
        explainer,
        texts: List[str],
        explanations: Dict[str, Any],
        methods: List[str] = ["shap", "lime"],
    ) -> Dict[str, Any]:
        """Evaluate explanation quality.
        
        Args:
            model: The model being explained.
            explainer: The explainer.
            texts: List of input texts.
            explanations: Dictionary containing explanations.
            methods: List of explanation methods to evaluate.
            
        Returns:
            Dictionary containing explanation evaluation metrics.
        """
        results = {}
        
        for method in methods:
            if method in explanations:
                # Faithfulness metrics
                faithfulness = self.metrics.faithfulness_metrics(
                    model, texts, explanations[method], method="deletion"
                )
                
                # Stability metrics
                stability = self.metrics.stability_metrics(
                    explainer, texts[:5], method=method  # Use subset for stability
                )
                
                results[method] = {
                    "faithfulness": faithfulness,
                    "stability": stability,
                }
        
        return results
