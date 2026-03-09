"""Explanation methods for NLP models using SHAP, LIME, and attention visualization."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
    ShapleyValueSampling,
)
from lime.lime_text import LimeTextExplainer
import shap
from transformers import AutoTokenizer

from ..models.bert import InterpretableBERT
from ..utils.device import get_device

warnings.filterwarnings("ignore", category=UserWarning)


class NLPExplainer:
    """Comprehensive explainer for NLP models using multiple methods."""
    
    def __init__(
        self,
        model: InterpretableBERT,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize NLP explainer.
        
        Args:
            model: The model to explain.
            tokenizer: Tokenizer for the model.
            device: Device to run explanations on.
        """
        self.model = model
        self.tokenizer = tokenizer or model.tokenizer
        self.device = device or get_device()
        self.model.eval()
    
    def explain_with_shap(
        self,
        texts: Union[str, List[str]],
        method: str = "kernel",
        max_samples: int = 100,
        background_samples: int = 50,
    ) -> Dict[str, Any]:
        """Explain predictions using SHAP.
        
        Args:
            texts: Text or list of texts to explain.
            method: SHAP method ('kernel', 'sampling', 'exact').
            max_samples: Maximum samples for approximation.
            background_samples: Number of background samples.
            
        Returns:
            Dictionary containing SHAP values and explanations.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Create background dataset
        if background_samples > 0:
            # Use random subset of texts as background
            background_texts = np.random.choice(
                texts, 
                min(background_samples, len(texts)), 
                replace=False
            ).tolist()
        else:
            background_texts = []
        
        def model_predict(texts_list):
            """Wrapper function for SHAP."""
            predictions, _ = self.model.predict(texts_list, return_probabilities=True)
            return np.array(predictions)
        
        try:
            if method == "kernel":
                explainer = shap.KernelExplainer(model_predict, background_texts)
                shap_values = explainer.shap_values(texts, nsamples=max_samples)
            elif method == "sampling":
                explainer = shap.SamplingExplainer(model_predict, background_texts)
                shap_values = explainer.shap_values(texts, nsamples=max_samples)
            else:
                raise ValueError(f"Unsupported SHAP method: {method}")
            
            return {
                "shap_values": shap_values,
                "method": method,
                "texts": texts,
            }
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return {"error": str(e)}
    
    def explain_with_lime(
        self,
        texts: Union[str, List[str]],
        num_features: int = 10,
        num_samples: int = 5000,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Explain predictions using LIME.
        
        Args:
            texts: Text or list of texts to explain.
            num_features: Number of features to highlight.
            num_samples: Number of samples for LIME.
            class_names: Names of classes.
            
        Returns:
            Dictionary containing LIME explanations.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        def model_predict_proba(texts_list):
            """Wrapper function for LIME."""
            _, probabilities = self.model.predict(texts_list, return_probabilities=True)
            return np.array(probabilities)
        
        explainer = LimeTextExplainer(
            class_names=class_names or [f"class_{i}" for i in range(self.model.num_labels)],
            random_state=42,
        )
        
        explanations = []
        for text in texts:
            try:
                explanation = explainer.explain_instance(
                    text,
                    model_predict_proba,
                    num_features=num_features,
                    num_samples=num_samples,
                )
                explanations.append(explanation)
            except Exception as e:
                print(f"LIME explanation failed for text: {e}")
                explanations.append(None)
        
        return {
            "explanations": explanations,
            "texts": texts,
            "num_features": num_features,
        }
    
    def explain_with_attention(
        self,
        texts: Union[str, List[str]],
        layer: int = -1,
        head: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explain predictions using attention weights.
        
        Args:
            texts: Text or list of texts to explain.
            layer: BERT layer to analyze (-1 for last layer).
            head: Specific attention head (None for average).
            
        Returns:
            Dictionary containing attention explanations.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        attention_weights = self.model.get_attention_weights(texts)
        
        explanations = []
        for i, text in enumerate(texts):
            # Get tokens for this text
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            
            # Get attention weights for specified layer
            layer_attention = attention_weights[i][layer]  # Shape: [num_heads, seq_len, seq_len]
            
            if head is not None:
                # Use specific head
                attention = layer_attention[head]
            else:
                # Average across heads
                attention = torch.mean(layer_attention, dim=0)
            
            # Get attention from [CLS] token to all other tokens
            cls_attention = attention[0, :].cpu().numpy()
            
            explanations.append({
                "tokens": tokens,
                "attention_weights": cls_attention,
                "layer": layer,
                "head": head,
            })
        
        return {
            "explanations": explanations,
            "texts": texts,
            "layer": layer,
            "head": head,
        }
    
    def explain_with_integrated_gradients(
        self,
        texts: Union[str, List[str]],
        baseline: str = "",
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """Explain predictions using Integrated Gradients.
        
        Args:
            texts: Text or list of texts to explain.
            baseline: Baseline text for IG computation.
            n_steps: Number of steps for integration.
            
        Returns:
            Dictionary containing IG explanations.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        def model_forward(input_ids, attention_mask):
            """Wrapper function for Captum."""
            outputs = self.model(input_ids, attention_mask)
            return outputs.logits
        
        ig = IntegratedGradients(model_forward)
        
        explanations = []
        for text in texts:
            try:
                # Tokenize text
                encoding = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                # Tokenize baseline
                baseline_encoding = self.tokenizer(
                    baseline,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                # Move to device
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                baseline_ids = baseline_encoding["input_ids"].to(self.device)
                baseline_mask = baseline_encoding["attention_mask"].to(self.device)
                
                # Compute attributions
                attributions = ig.attribute(
                    input_ids,
                    baselines=baseline_ids,
                    additional_forward_args=(attention_mask,),
                    n_steps=n_steps,
                )
                
                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                explanations.append({
                    "tokens": tokens,
                    "attributions": attributions[0].cpu().numpy(),
                    "text": text,
                })
                
            except Exception as e:
                print(f"IG explanation failed for text: {e}")
                explanations.append(None)
        
        return {
            "explanations": explanations,
            "texts": texts,
            "n_steps": n_steps,
        }
    
    def explain_with_saliency(
        self,
        texts: Union[str, List[str]],
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explain predictions using saliency maps.
        
        Args:
            texts: Text or list of texts to explain.
            target_class: Target class for explanation (None for predicted class).
            
        Returns:
            Dictionary containing saliency explanations.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        def model_forward(input_ids, attention_mask):
            """Wrapper function for Captum."""
            outputs = self.model(input_ids, attention_mask)
            return outputs.logits
        
        saliency = Saliency(model_forward)
        
        explanations = []
        for text in texts:
            try:
                # Tokenize text
                encoding = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                
                # Get prediction
                predictions, _ = self.model.predict([text], return_probabilities=True)
                predicted_class = predictions[0]
                
                # Use predicted class if target not specified
                if target_class is None:
                    target_class = predicted_class
                
                # Compute saliency
                attributions = saliency.attribute(
                    input_ids,
                    additional_forward_args=(attention_mask,),
                    target=target_class,
                )
                
                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                explanations.append({
                    "tokens": tokens,
                    "attributions": attributions[0].cpu().numpy(),
                    "target_class": target_class,
                    "predicted_class": predicted_class,
                })
                
            except Exception as e:
                print(f"Saliency explanation failed for text: {e}")
                explanations.append(None)
        
        return {
            "explanations": explanations,
            "texts": texts,
            "target_class": target_class,
        }
    
    def compare_explanations(
        self,
        text: str,
        methods: List[str] = ["shap", "lime", "attention", "ig"],
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare multiple explanation methods for a single text.
        
        Args:
            text: Text to explain.
            methods: List of explanation methods to use.
            **kwargs: Additional arguments for explanation methods.
            
        Returns:
            Dictionary containing all explanations.
        """
        explanations = {}
        
        if "shap" in methods:
            explanations["shap"] = self.explain_with_shap(text, **kwargs.get("shap", {}))
        
        if "lime" in methods:
            explanations["lime"] = self.explain_with_lime(text, **kwargs.get("lime", {}))
        
        if "attention" in methods:
            explanations["attention"] = self.explain_with_attention(text, **kwargs.get("attention", {}))
        
        if "ig" in methods:
            explanations["ig"] = self.explain_with_integrated_gradients(text, **kwargs.get("ig", {}))
        
        if "saliency" in methods:
            explanations["saliency"] = self.explain_with_saliency(text, **kwargs.get("saliency", {}))
        
        return explanations
