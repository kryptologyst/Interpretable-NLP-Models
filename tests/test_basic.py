"""Basic tests for the interpretable NLP models package."""

import pytest
import torch
import numpy as np

from src.data.loader import NLPDataset, create_synthetic_dataset
from src.models.bert import InterpretableBERT
from src.utils.device import set_seed, get_device


class TestNLPDataset:
    """Test cases for NLPDataset."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        texts = ["This is a test", "Another test text", "Third test"]
        labels = [0, 1, 0]
        label_names = ["class_0", "class_1"]
        
        dataset = NLPDataset(texts, labels, label_names)
        
        assert len(dataset.texts) == 3
        assert len(dataset.labels) == 3
        assert dataset.n_samples == 3
        assert dataset.n_classes == 2
    
    def test_dataset_split(self):
        """Test dataset splitting."""
        texts = ["text"] * 100
        labels = list(range(100))
        label_names = [f"class_{i}" for i in range(100)]
        
        dataset = NLPDataset(texts, labels, label_names)
        train, val, test = dataset.split(test_size=0.2, val_size=0.1)
        
        assert len(train.texts) + len(val.texts) + len(test.texts) == 100
        assert len(test.texts) == 20  # 20% of 100
        assert len(val.texts) == 10   # 10% of 100
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(n_samples=50, n_classes=3)
        
        assert len(dataset.texts) == 50
        assert dataset.n_classes == 3
        assert all(isinstance(text, str) for text in dataset.texts)
        assert all(isinstance(label, int) for label in dataset.labels)


class TestInterpretableBERT:
    """Test cases for InterpretableBERT."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        model = InterpretableBERT(
            model_name="bert-base-uncased",
            num_labels=5,
        )
        
        assert model.num_labels == 5
        assert model.model_name == "bert-base-uncased"
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "bert")
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = InterpretableBERT(
            model_name="bert-base-uncased",
            num_labels=3,
        )
        
        texts = ["This is a test", "Another test"]
        predictions = model.predict(texts)
        
        assert len(predictions) == 2
        assert all(isinstance(pred, int) for pred in predictions)
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_model_prediction_with_probabilities(self):
        """Test model prediction with probabilities."""
        model = InterpretableBERT(
            model_name="bert-base-uncased",
            num_labels=3,
        )
        
        texts = ["This is a test"]
        predictions, probabilities = model.predict(texts, return_probabilities=True)
        
        assert len(predictions) == 1
        assert len(probabilities) == 1
        assert len(probabilities[0]) == 3
        assert abs(sum(probabilities[0]) - 1.0) < 1e-6  # Probabilities sum to 1


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        assert torch.initial_seed() is not None
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # Create synthetic dataset
        dataset = create_synthetic_dataset(n_samples=10, n_classes=3)
        
        # Create model
        model = InterpretableBERT(
            model_name="bert-base-uncased",
            num_labels=3,
        )
        
        # Make predictions
        predictions = model.predict(dataset.texts)
        
        assert len(predictions) == 10
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Create model
        model = InterpretableBERT(
            model_name="bert-base-uncased",
            num_labels=3,
        )
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_model(temp_dir)
            
            # Check files exist
            assert os.path.exists(os.path.join(temp_dir, "model.pt"))
            assert os.path.exists(os.path.join(temp_dir, "config.pt"))
            
            # Load model
            loaded_model = InterpretableBERT.load_model(temp_dir)
            
            assert loaded_model.num_labels == model.num_labels
            assert loaded_model.model_name == model.model_name


if __name__ == "__main__":
    pytest.main([__file__])
