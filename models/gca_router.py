import torch
import torch.nn as nn
import torch.nn.functional as F

class GCARouter(nn.Module):
    """
    GCA (Gated Cross-Attention) Router.
    
    Acts as the entry point for inference:
    Takes pre-extracted DINOv2 (visual) and CLIP (text) features.
    Fuses them using Cross-Attention.
    Outputs a routing decision:
      - 0 (Simple/No-conflict image) -> S1 Fast Channel (Absolute Score MLP)
      - 1 (Complex/High-conflict image) -> S2 Slow Channel (Wake up LLM & Q-Probe)
    """
    def __init__(self, visual_dim: int = 768, text_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        # Projections to common space
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross Attention (Query=Text, Key=Value=Visual)
        # ？？？？？？？ 
        # Exact attention setup is a placeholder. MultiheadAttention is standard.
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # S1 Fast Channel (Absolute Score MLP)
        self.mlp_s1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Output score 0-100
        )
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        """
        Inputs:
            visual_features: (B, N_v, visual_dim) or (B, visual_dim)
            text_features: (B, N_t, text_dim) or (B, text_dim)
        """
        # Ensure 3D shape for attention (Batch, Sequence, Dim)
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
            
        # Project
        v_emb = self.v_proj(visual_features)
        t_emb = self.t_proj(text_features)
        
        # Cross Attention: Text queries Visual context
        attn_out, _ = self.cross_attn(query=t_emb, key=v_emb, value=v_emb)
        
        # Aggregate (e.g., mean pooling over sequence)
        fused_features = attn_out.mean(dim=1)
        
        # Gating Decision (0 = S1, 1 = S2)
        p_s2 = self.gate(fused_features)
        
        # Fast Channel Score (S1) computation
        s1_score = self.mlp_s1(fused_features)
        s1_score = torch.clamp(s1_score, 0.0, 100.0)
        
        return {
            "p_s2": p_s2,         # Probability of routing to S2 Slow Channel
            "s1_score": s1_score, # Absolute score if routed to S1 Fast Channel
            "fused_features": fused_features
        }

# Example instantiation
if __name__ == "__main__":
    router = GCARouter()
    v_feat = torch.randn(2, 768)
    t_feat = torch.randn(2, 512)
    out = router(v_feat, t_feat)
    print("P(S2):", out["p_s2"])
    print("S1 Score:", out["s1_score"])
