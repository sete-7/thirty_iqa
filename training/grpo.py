import torch

def fast_grpo_reward(base_reward: torch.Tensor, 
                     S_d: torch.Tensor, 
                     answer_length: torch.Tensor, 
                     length_threshold: int = 200, 
                     penalty_factor: float = 0.1) -> torch.Tensor:
    """
    FAST-GRPO Reward Calculation with Adaptive Length Penalty.
    
    If the intrinsic difficulty S_d of the image is low and the answer is excessively long,
    the model receives a negative reward penalty to discourage verbosity on simple images.
    
    # ？？？？？？？
    # The explicit formula for the continuous penalty mapping based on S_d and length 
    # is not detailed. A linear threshold-based proxy is used here.
    """
    # Create penalty mask: 1 where S_d is low (< threshold) and length is high (> threshold)
    # ？？？？？？？ Threshold for S_d
    S_d_threshold = 0.5 
    
    penalty_mask = (S_d < S_d_threshold) & (answer_length > length_threshold)
    
    # Calculate adaptive penalty (proportional to how much it exceeds the length)
    excess_length = (answer_length - length_threshold).float()
    adaptive_penalty = penalty_factor * excess_length * penalty_mask.float()
    
    # Final reward is base reward minus the adaptive penalty
    final_reward = base_reward - adaptive_penalty
    
    return final_reward
