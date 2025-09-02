from transformers import AutoModelForCausalLM, AutoConfig, Qwen3ForCausalLM
from verl.models.entropy_embeddings import EntropyEmbeddingProjection
import torch.nn as nn
from vllm.model_executor.models.transformers import TransformersForCausalLM
from transformers import Qwen3ForCausalLM
import torch
from torch._dynamo import allow_in_graph, disable, graph_break

 
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

        # Add your extra embedding
        self.entropy_embedding = EntropyEmbeddingProjection(hf_config.hidden_size)


        # Hook into embedding layer
        emb = self.model.model.embed_tokens

        dtype = self.entropy_embedding.entropy_projection[0].weight.dtype
        device = self.entropy_embedding.entropy_projection[0].weight.device
        emb._last_entropy = torch.zeros((1,1), dtype=dtype, device=device)
        
        orig_emb_forward = emb.forward
        
        def new_emb_forward(*args, **kwargs):
            x = orig_emb_forward(*args, **kwargs)
            x = x + self.entropy_embedding(emb._last_entropy)
            return x         
        emb.forward = new_emb_forward
        
        #Forward hook        
        orig_forward = self.model.forward
        def new_forward(*args, **kwargs):
            outputs = orig_forward(*args, **kwargs) #(?, 151936)

        # Extract logits robustly
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = None

            if logits is not None:
                probs = torch.softmax(logits, dim=-1)
                entropy_seq = -(probs * torch.log(probs + 1e-8)).sum()  # (?, ) Apparently 32768 if sum(-1) or else 1
                emb._last_entropy = entropy_seq.reshape(-1,1).detach().to(emb._last_entropy.device)
            return outputs
        self.model.forward = new_forward
        
        print("[ENTROPY] USING WRAPPER, FINISHED INIT")

        
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

        #DEBUGGING
        entropy_keys = [k for k in provided_keys if k.startswith("entropy_embedding.")]
        if len(entropy_keys)>0:
            for k in entropy_keys:
                print(f"[WEIGHT LOADER] Found {k}")
        else:
            print("[WEIGHT LOADER] Did not find entropy embedding weights")


        embed_keys = [k for k in provided_keys if k.startswith("model.embed_tokens.")]
        if len(embed_keys)>0:
            for k in embed_keys:
                print(f"[WEIGHT LOADER] Found {k}")
        else:
            print("[WEIGHT LOADER] Did not find any embedding weights")



        # If entropy params are missing, inject the module's current weights
        if not any(k.startswith("entropy_embedding.") for k in provided_keys):
            extra = []
            for k, v in self.entropy_embedding.state_dict().items():
                if k not in provided_keys:
                    extra.append((f"entropy_embedding.{k}", v.to(next(self.parameters()).device)))
            params_iter.extend(extra)

        # Now call parent loader with the combined list
        return super().load_weights(params_iter, *args, **kwargs)


class EmbedWrapper(nn.Module):
    def __init__(self, token_embedding, ent_embedding):
        super().__init__()
        self.ent_emb = ent_embedding
        self.orig_emb = token_embedding
        self.register_buffer("_last_entropy", torch.zeros((1,1,1)))
        
    def __getattr__(self, name):
        # defer unknown attributes to the original embedding
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.orig_emb, name)
    
    def forward(self, input):
        out = self.orig_emb(input)

        #If shape is not equal (ie due to the prompt) then start from zero padding. 
        if self._last_entropy.shape[1] != out.shape[1]:
            # with torch.no_grad():
                # logits = self._model_forward(inputs_embeds=out).last_hidden_state
                # entropies = self.compute_prompt_entropies(logits)
                # self._last_entropy = entropies
            self._last_entropy = torch.zeros((out.shape[0],out.shape[1],1), dtype=out.dtype, device=out.device)
            
            print("[FEEDBACK TENSOR] SHAPE DIFFERENCE: RESETTING")

        add = self.ent_emb(self._last_entropy.detach())                # keep module on correct device once, outside the hook
        out = out + add
            
        print("[FEEDBACK TENSOR] Last entropy: ", self._last_entropy)
        print("[FEEDBACK TENSOR] embed forward, entropy: ", add.shape)
        print("[FEEDBACK TENSOR] embed forward, token: ",out.shape)
        print("[FEEDBACK TENSOR] entropy shape: ", self._last_entropy.shape)
        return out 

    def compute_prompt_entropies(self, logits):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropies = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-1, keepdim=True).detach()   # (B, T)
            return entropies

class EntropyEnhancedWrapperHF(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        def lm_head_hook(module, inputs, logits):
            # logits: (B, T, V)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                ent_seq = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True).detach()   # (B, T)
                self.model.embed_tokens._last_entropy = ent_seq

                # last = ent_seq[:, -1:].unsqueeze(-1).detach()                       # (B, 1, 1)
                # self.model.embed_tokens._last_entropy = torch.cat((self.model.embed_tokens._last_entropy, last),dim=1)

            return logits
        self.lm_head.register_forward_hook(lm_head_hook)

