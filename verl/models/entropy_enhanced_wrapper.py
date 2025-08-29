from transformers import AutoModelForCausalLM, AutoConfig, Qwen3ForCausalLM
from verl.models.entropy_embeddings import EntropyEmbeddingProjection
import torch.nn as nn
from vllm.model_executor.models.transformers import TransformersForCausalLM
from transformers import Qwen3ForCausalLM
import torch
from torch._dynamo import allow_in_graph
from torch._dynamo import disable



class EntropyEnhancedEmbedding(nn.Module):
    def __init__(self, base_emb, entropy_module):
        super().__init__()
        self.base_emb = base_emb
        self._entropy_module = entropy_module
        self._last_entropy = None

    def set_entropy(self, past_entropy):
        self._last_entropy = past_entropy

    def forward(self, input_ids):
        embeds = self.base_emb(input_ids)
        # if self._last_entropy is not None:
            # embeds = embeds + self.entropy_module(self._last_entropy)
        embeds = torch.rand_like(embeds)*8-4
        return embeds

class EntropyAugmentedEmbedding(nn.Module):
    def __init__(self, base_emb, entropy_emb):
        super().__init__()
        self.base_emb = base_emb
        self.entropy_emb = entropy_emb
        self._last_entropy = None
    def set_entropy(self, past_entropy): self._last_entropy = past_entropy
    def forward(self, input_ids):
        embs = self.base_emb(input_ids)
        # if self._last_entropy is not None:
            # x = x + self.entropy_emb(self._last_entropy)
        x = torch.rand_like(embs)*8-4
        return x
    
    
class EntropyEnhancedModelWrapper(TransformersForCausalLM):
    def __init__(self, *, vllm_config=None, config=None, prefix=None, **kwargs):
        """
        Support both vLLM ≤0.6 ABI styles:
        - V1 path: __init__(vllm_config=...)
        - Old path: __init__(config=..., prefix=..., ...)
        """

        hf_config = None

        if vllm_config is not None:
            # Newer path: vllm_config carries the HF config inside
            super().__init__(vllm_config=vllm_config)
            # In some releases this is vllm_config.model_config, in others model_config.hf_config
            hf_config = getattr(vllm_config, "model_config", None)
            if hf_config is not None and hasattr(hf_config, "hf_config"):
                hf_config = hf_config.hf_config

        elif config is not None:
            # Older path: config is passed directly
            super().__init__()  # old base ctor takes no args
            hf_config = config

        else:
            raise ValueError("EntropyEnhancedModelWrapper requires either vllm_config or config.")

        if hf_config is None:
            raise ValueError("Could not resolve HuggingFace config from vllm_config/config.")

        # Build the HuggingFace Qwen3 model
        # self.transformer = Qwen3ForCausalLM(hf_config)

        # Add your extra embedding
        self.entropy_embedding = EntropyEmbeddingProjection(hf_config.hidden_size)
        # self.entropy_embedding = EntropyEmbeddingProjection(hf_config.hidden_size)
        
        # class EntropyWrapper(nn.Module):
        #     def __init__(self, parent):
        #         super().__init__()
        #         # self.base_emb = base_emb
        #         # self.entropy_module = entropy_module
        #         self.parent = parent
        #     def forward(self, input_ids):
        #         embeds = self.parent.model.model.embed_tokens(input_ids)
        #         # if self.parent._last_entropy is not None:
        #             # embeds = embeds + self.entropy_module(self.parent._last_entropy)
        #         embeds = torch.rand_like(embeds)*8-4
        #         return embeds
        
        self._last_entropy = None
        emb = self.model.model.embed_tokens
        orig_forward = emb.forward
        def new_forward_fn(*args, **kwargs):
            x = orig_forward(*args, **kwargs)
            if self._last_entropy is not None:
                x = x + self.entropy_embedding(self._last_entropy)
            return x         
        emb.forward = new_forward_fn        
        
        print("[ENTROPY] USING WRAPPER, FINISHED INIT")
        print(f"[ENTROPY] INPUT EMBEDDING MODULE: {self.model.get_input_embeddings}")


    def set_entropy(self, past_entropy):
        """Store entropy input for the next forward pass."""
        self._last_entropy = past_entropy
        
    def load_weights(self, params, *args, **kwargs):
        """
        Override to handle entropy_embedding.* specially.

        - On checkpoint load: vLLM passes an iterable of (name, tensor) from the HF ckpt.
        These won't include entropy keys. We inject the wrapper's own initialized
        entropy weights so vLLM's strict checker is satisfied.
        - On optimizer update: params is usually a dict with *all* keys (including entropy).
        In that case, just pass them through so updates are applied.
        """

        # Normalize into an iterable of (name, tensor)
        if isinstance(params, dict):
            params_iter = list(params.items())
        else:
            params_iter = list(params)

        # Collect the provided keys
        provided_keys = {k for k, _ in params_iter}

        # If entropy params are missing, inject the module's current weights
        if not any(k.startswith("entropy_embedding.") for k in provided_keys):
            extra = []
            for k, v in self.entropy_embedding.state_dict().items():
                if k not in provided_keys:
                    extra.append((f"entropy_embedding.{k}", v.to(next(self.parameters()).device)))
            params_iter.extend(extra)

        # Now call parent loader with the combined list
        return super().load_weights(params_iter, *args, **kwargs)

