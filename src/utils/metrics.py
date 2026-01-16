
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
    
    # Balanced Accuracy (Prompt 5)
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    bal_acc = balanced_accuracy_score(targets, preds)
    
    # Per-class F1
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    results = {
        'val_acc': acc,
        'val_bal_acc': bal_acc,
        'val_f1': macro_f1,
        'cm': cm # Return matrix object if needed, or string?
    }
    
    # Add per-class F1 to flat dict
    for i, f1 in enumerate(per_class_f1):
        if i < num_classes:
            results[f'val_f1_class_{i}'] = f1
            
    return results
