
import unittest
import torch
import shutil
import os
from omegaconf import OmegaConf
from src.methods.erm import ERM
from src.models.resnet1d import ResNet1d
from torch.utils.data import DataLoader, TensorDataset

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # Temp dir for outputs
        self.output_dir = "tests/temp_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_model_forward(self):
        """Test ResNet1d forward pass with random input"""
        model = ResNet1d(input_channels=12, num_classes=7)
        x = torch.randn(2, 12, 1000) # Batch 2, 12 leads, 1000 samples
        logits = model(x)
        self.assertEqual(logits.shape, (2, 7))

    def test_erm_step(self):
        """Test ERM training step"""
        model = ResNet1d(input_channels=12, num_classes=7)
        method = ERM(model, num_classes=7)
        optimizer = method.configure_optimizers(lr=1e-3, weight_decay=0.0)
        
        x = torch.randn(4, 12, 1000)
        y = torch.tensor([0, 1, 2, 3])
        
        # Train step
        optimizer.zero_grad()
        out = method.training_step((x, y), 0)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        
    def test_full_epoch_simulation(self):
        """Simulate one epoch of training loop"""
        # Mock Data
        x = torch.randn(10, 12, 1000)
        y = torch.randint(0, 7, (10,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=2)
        
        model = ResNet1d(input_channels=12, num_classes=7)
        method = ERM(model, num_classes=7)
        optimizer = method.configure_optimizers(lr=1e-3, weight_decay=0.0)
        
        method.train()
        for batch in loader:
            optimizer.zero_grad()
            out = method.training_step(batch, 0)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
