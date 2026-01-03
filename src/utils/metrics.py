
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

def calculate_metrics(preds, targets, num_classes=7):
    """
    Calculate metrics for classification.
    Args:
        preds (torch.Tensor or np.array): Predictions
        targets (torch.Tensor or np.array): Ground Truth
    Returns:
        dict: {'acc': float, 'macro_f1': float, 'report': str}
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    # Per class report (optional, maybe too verbose for log)
    # report = classification_report(targets, preds, output_dict=True, zero_division=0)
    
    return {
        'val_acc': acc,
        'val_f1': macro_f1
    }
