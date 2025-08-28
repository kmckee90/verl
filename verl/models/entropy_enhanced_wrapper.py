import torch
import torch.nn as nn
from verl.models.entropy_embeddings import EntropyEmbeddingProjection

class EntropyEnhancedModelWrapper(nn.Module):
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.entropy_embedding = EntropyEmbeddingProjection(hidden_size)

    def __getattr__(self, name):
        # Delegate all unknown attributes to base_model
        # Check if base_model is in our modules (where nn.Module stores submodules)
        if hasattr(self, '_modules') and 'base_model' in self._modules:
            base_model = self._modules['base_model']
            try:
                return getattr(base_model, name)
            except AttributeError:
                pass
        
        # If we get here, neither wrapper nor base_model has the attribute
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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

            inputs_embeds = inputs_embeds + entropy_embeds
        else:
            print(f"[ENTROPY_DEBUG] past_entropy is None - entropy embeddings NOT applied")

        # Forward through rest of model
        return self.base_model(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              **kwargs)
    
