import math
import torch

def apply_uado_decay(optimizer: torch.optim.Optimizer, 
                     batch_scores: torch.Tensor, 
                     eta: float = 0.1):
    """
    UADO (Uncertainty-Aware Dropout/Decay Optimizer) Mechanism.
    
    Before gradient update, calculates the score variance of the same batch/sample.
    Multiplies the learning rate by a decay factor: exp(-eta * variance).
    This prevents extreme samples from causing gradient explosion.
    
    # ？？？？？？？
    # The exact grouping for variance (e.g., per sample across multiple completions vs whole batch)
    # is inferred here as the variance of scores within the current batch.
    """
    # Calculate variance of scores in the batch
    variance = torch.var(batch_scores, unbiased=False)
    
    # Compute decay factor: exp(-η * var)
    decay_factor = math.exp(-eta * variance.item())
    
    # Apply decay to optimizer learning rate
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
        
        # Scale the current scheduled LR by the UADO decay factor
        param_group['lr'] = param_group['lr'] * decay_factor
        
    return decay_factor
