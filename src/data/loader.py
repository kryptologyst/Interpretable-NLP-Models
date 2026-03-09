"""Data loading and preprocessing utilities for NLP tasks."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class NLPDataset:
    """Wrapper class for NLP datasets with metadata and preprocessing."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        label_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize NLP dataset.
        
        Args:
            texts: List of text samples.
            labels: List of corresponding labels.
            label_names: List of label names.
            metadata: Additional metadata about the dataset.
        """
        self.texts = texts
        self.labels = labels
        self.label_names = label_names
        self.metadata = metadata or {}
        
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        self.n_samples = len(texts)
        self.n_classes = len(set(labels))
        
        # Add basic statistics to metadata
        self.metadata.update({
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "class_distribution": self._get_class_distribution(),
            "avg_text_length": np.mean([len(text.split()) for text in texts]),
        })
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return {self.label_names[label]: int(count) for label, count in zip(unique_labels, counts)}
    
    def get_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets."""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        weights = {}
        
        for label, count in zip(unique_labels, counts):
            weights[int(label)] = total_samples / (self.n_classes * count)
        
        return weights
    
    def split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple["NLPDataset", "NLPDataset", "NLPDataset"]:
        """Split dataset into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for test set.
            val_size: Proportion of data for validation set.
            random_state: Random seed for reproducibility.
            stratify: Whether to stratify splits by labels.
            
        Returns:
            Tuple of (train, val, test) NLPDataset objects.
        """
        stratify_labels = self.labels if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.texts,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp,
        )
        
        # Create new dataset objects
        train_dataset = NLPDataset(X_train, y_train, self.label_names, self.metadata.copy())
        val_dataset = NLPDataset(X_val, y_val, self.label_names, self.metadata.copy())
        test_dataset = NLPDataset(X_test, y_test, self.label_names, self.metadata.copy())
        
        return train_dataset, val_dataset, test_dataset
    
    def save_metadata(self, filepath: str) -> None:
        """Save dataset metadata to JSON file.
        
        Args:
            filepath: Path to save metadata.
        """
        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.
        
        Returns:
            pd.DataFrame: Dataset as DataFrame with 'text' and 'label' columns.
        """
        return pd.DataFrame({
            "text": self.texts,
            "label": self.labels,
            "label_name": [self.label_names[label] for label in self.labels],
        })


def load_20newsgroups(
    subset: str = "all",
    remove: Tuple[str, ...] = ("headers", "footers", "quotes"),
    categories: Optional[List[str]] = None,
) -> NLPDataset:
    """Load 20 Newsgroups dataset.
    
    Args:
        subset: Subset to load ('train', 'test', or 'all').
        remove: Parts to remove from documents.
        categories: Specific categories to load (None for all).
        
    Returns:
        NLPDataset: Loaded dataset.
    """
    newsgroups = fetch_20newsgroups(
        subset=subset,
        remove=remove,
        categories=categories,
    )
    
    metadata = {
        "dataset_name": "20newsgroups",
        "subset": subset,
        "removed_parts": list(remove),
        "categories": categories or "all",
    }
    
    return NLPDataset(
        texts=newsgroups.data,
        labels=newsgroups.target,
        label_names=newsgroups.target_names,
        metadata=metadata,
    )


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_classes: int = 5,
    text_length_range: Tuple[int, int] = (10, 100),
    random_state: int = 42,
) -> NLPDataset:
    """Create synthetic text classification dataset for testing.
    
    Args:
        n_samples: Number of samples to generate.
        n_classes: Number of classes.
        text_length_range: Range of text lengths (min, max words).
        random_state: Random seed.
        
    Returns:
        NLPDataset: Synthetic dataset.
    """
    np.random.seed(random_state)
    
    # Define topic templates for each class
    topics = [
        "technology computer software programming",
        "sports football basketball baseball",
        "politics government election democracy",
        "science research discovery innovation",
        "entertainment movie music television",
    ]
    
    # Generate texts
    texts = []
    labels = []
    
    for i in range(n_samples):
        class_idx = i % n_classes
        topic_words = topics[class_idx].split()
        
        # Generate random text length
        text_length = np.random.randint(text_length_range[0], text_length_range[1] + 1)
        
        # Generate text by sampling from topic words and adding noise
        text_words = []
        for _ in range(text_length):
            if np.random.random() < 0.7:  # 70% chance to use topic word
                word = np.random.choice(topic_words)
            else:  # 30% chance to use random word
                word = f"word{np.random.randint(1000)}"
            text_words.append(word)
        
        texts.append(" ".join(text_words))
        labels.append(class_idx)
    
    label_names = [f"class_{i}" for i in range(n_classes)]
    
    metadata = {
        "dataset_name": "synthetic",
        "n_samples": n_samples,
        "n_classes": n_classes,
        "text_length_range": text_length_range,
        "random_state": random_state,
    }
    
    return NLPDataset(texts, labels, label_names, metadata)


class TextTokenizer:
    """Text tokenizer wrapper for various tokenizers."""
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """Initialize tokenizer.
        
        Args:
            tokenizer_name: Name of the tokenizer to use.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate sequences.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts.
        
        Args:
            texts: Text or list of texts to tokenize.
            return_tensors: Format of returned tensors.
            
        Returns:
            Dict containing tokenized inputs.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors,
        )
        
        return encodings
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.
            
        Returns:
            List of decoded texts.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
