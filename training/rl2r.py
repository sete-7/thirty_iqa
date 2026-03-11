import torch
import torch.nn as nn
import torch.nn.functional as F

class RL2RLoss(nn.Module):
    """
    RL2R Algorithm: Converts model output quality scores into a 1D Gaussian distribution 
    (mean and variance) and implements the Bradley-Terry relative ranking loss.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                pred_mean_i: torch.Tensor, pred_var_i: torch.Tensor, 
                pred_mean_j: torch.Tensor, pred_var_j: torch.Tensor, 
                y_true_ij: torch.Tensor):
        """
        # ？？？？？？？
        # The exact implementation of sampling from the Gaussian and integrating it 
        # into the Bradley-Terry loss is not explicitly provided. 
        # Here we provide a continuous BT loss proxy using Gaussian parameters.
        
        Args:
            pred_mean_i, pred_var_i: Gaussian params for image i
            pred_mean_j, pred_var_j: Gaussian params for image j
            y_true_ij: 1 if i > j, 0 if i < j, 0.5 if tie.
        """
        # Standard BT: P(i > j) = sigmoid(score_i - score_j)
        # For Gaussians, the difference D = S_i - S_j is also Gaussian:
        # D ~ N(mean_i - mean_j, var_i + var_j)
        
        diff_mean = pred_mean_i - pred_mean_j
        diff_var = pred_var_i + pred_var_j + 1e-6
        
        # ？？？？？？？
        # Prob(i > j) considering the Gaussian distribution.
        # Approximation: scale the difference mean by the standard deviation.
        c = np.sqrt(math.pi / 8.0) # commonly used probit approximation scaling
        logits = diff_mean / torch.sqrt(1.0 + c**2 * diff_var)
        
        # log_loss = - [y * log(sigmoid(logits)) + (1-y) * log(1-sigmoid(logits))]
        loss = F.binary_cross_entropy_with_logits(logits, y_true_ij)
        
        return loss
    
import math
import numpy as np
