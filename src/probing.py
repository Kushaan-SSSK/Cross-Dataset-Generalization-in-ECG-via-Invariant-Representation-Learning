
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import ECGDataset
import logging

log = logging.getLogger(__name__)

class ProbingSuite:
    """
    Analyzes learned representations (Z, Z_c, Z_s).
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def extract_features(self, loader, method_type='erm'):
        """
        Extract features from the dataset.
        Args:
           method_type: 'erm' (Z=full), 'disentangled' (Z_c, Z_s)
        """
        features_list = []
        labels_list = []
        domains_list = []
        
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                x, y, d = batch[:3]
                
                # Get representation
                # DANN/ERM: model(x) -> features
                # Disentangled: model(x) -> full features -> split later
                
                feats = self.model(x) 
                # Note: For ResNet1d we modified it to return features before FC?
                # Or wait, for Disentangled/DANN we might have wrapped it.
                # Assuming 'self.model' here is the BACKBONE (ResNet1d_Encoder).
                
                features_list.append(feats.cpu().numpy())
                labels_list.append(y.cpu().numpy())
                domains_list.append(d.cpu().numpy())
                
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        domains = np.concatenate(domains_list)
        
        return features, labels, domains

    def run_probe(self, features_train, y_train, features_test, y_test, description="Probe"):
        """
        Train a linear classifier on frozen features.
        """
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        clf.fit(features_train, y_train)
        
        preds = clf.predict(features_test)
        acc = accuracy_score(y_test, preds)
        
        log.info(f"[{description}] Acc: {acc:.4f}")
        return acc

    def analyze(self, train_loader, val_loader, methods=['erm']):
        """
        Main analysis loop.
        """
        # Extract
        log.info("Extracting Train Features...")
        X_train, y_train, d_train = self.extract_features(train_loader)
        log.info("Extracting Val Features...")
        X_val, y_val, d_val = self.extract_features(val_loader)
        
        results = {}
        
        # 1. Task Predictability (Linear Probe)
        # Does Z predict Disease?
        task_acc = self.run_probe(X_train, y_train, X_val, y_val, "Task Probe (Z -> Y)")
        results['task_acc'] = task_acc
        
        # 2. Domain Predictability (Linear Probe)
        # Does Z predict Hospital? (Ideally NO for Invariant Z)
        domain_acc = self.run_probe(X_train, d_train, X_val, d_val, "Domain Probe (Z -> D)")
        results['domain_acc'] = domain_acc
        
        return results
