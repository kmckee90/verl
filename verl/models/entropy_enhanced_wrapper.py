import torch
import torch.nn as nn
from verl.models.entropy_embeddings import EntropyEmbeddingProjection

class EntropyEnhancedModelWrapper(nn.Module):
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.entropy_embedding = EntropyEmbeddingProjection(hidden_size)
        
        # Track entropy embedding statistics for logging
        self.entropy_embedding_norms = []
        self.entropy_application_count = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_entropy=None, **kwargs):
        # Get embeddings from base model
        if hasattr(self.base_model, 'embed_tokens'):
            inputs_embeds = self.base_model.embed_tokens(input_ids)
        else:
            # For HF models
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Add entropy information if available
        if past_entropy is not None:
            entropy_embeds = self.entropy_embedding(past_entropy)
            
            # Debug logging for entropy embeddings
            entropy_norm = entropy_embeds.norm().item()
            print(f"[ENTROPY_DEBUG] past_entropy shape: {past_entropy.shape}")
            print(f"[ENTROPY_DEBUG] past_entropy non-zero elements: {(past_entropy != 0).sum().item()}/{past_entropy.numel()}")
            print(f"[ENTROPY_DEBUG] past_entropy range: [{past_entropy.min().item():.6f}, {past_entropy.max().item():.6f}]")
            print(f"[ENTROPY_DEBUG] entropy_embeds shape: {entropy_embeds.shape}")
            print(f"[ENTROPY_DEBUG] entropy_embeds norm: {entropy_norm:.6f}")
            print(f"[ENTROPY_DEBUG] entropy_embeds mean abs: {entropy_embeds.abs().mean().item():.6f}")
            
            # Track for wandb logging
            self.entropy_embedding_norms.append(entropy_norm)
            self.entropy_application_count += 1
            
            inputs_embeds = inputs_embeds + entropy_embeds
        else:
            print(f"[ENTROPY_DEBUG] past_entropy is None - entropy embeddings NOT applied")

        # Forward through rest of model
        return self.base_model(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              **kwargs)
    
    def get_entropy_metrics(self):
        """Get and reset entropy embedding metrics for logging."""
        if not self.entropy_embedding_norms:
            return {}
        
        import torch
        norms_tensor = torch.tensor(self.entropy_embedding_norms)
        metrics = {
            "entropy/embedding_norm_mean": norms_tensor.mean().item(),
            "entropy/embedding_norm_std": norms_tensor.std().item(),
            "entropy/embedding_norm_max": norms_tensor.max().item(),
            "entropy/embedding_norm_min": norms_tensor.min().item(),
            "entropy/application_count": self.entropy_application_count,
        }
        
        # Reset for next collection period
        self.entropy_embedding_norms.clear()
        self.entropy_application_count = 0
        
        return metrics

