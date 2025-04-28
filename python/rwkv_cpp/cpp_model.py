from typing import Optional, Tuple, Dict, Any
import torch
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model

class CppModelConfig(PretrainedConfig):
    """Configuration class for the C++ model wrapper."""
    model_type: str = "cpp_model"
    
    def __init__(
        self,
        model_path: str = "",
        gpu_layers: int = 0,
        state_path: str = "",
        **kwargs
    ):
        self.model_path = model_path
        self.state_path = state_path
        self.gpu_layers = gpu_layers
        super().__init__(**kwargs)


class CppModelForCausalLM(PreTrainedModel, GenerationMixin):
    """Minimal implementation of a C++ model wrapper compatible with HuggingFace's generate API."""
    
    config_class = CppModelConfig
    
    def __init__(self, config: CppModelConfig):
        super().__init__(config)
        
        # Load the C++ model
        self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        self.model = rwkv_cpp_model.RWKVModel(
            self.library, 
            config.model_path, 
            gpu_layer_count=config.gpu_layers
        )
        
        # Current state of the model (will be updated during generation)
        self.current_state = None
        
        # Set vocab size from loaded model
        config.vocab_size = self.model.n_vocab
        self.config = config
        
        # Device tracking (for compatibility)
        self._device = torch.device("cpu")
        
        # Add a dummy parameter to make the model work with Transformers
        self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))

    def initialize_with_tuned_state(self, state_path):
        """
        Initialize the model with pre-tuned state tensors from a state dictionary.
        
        Parameters:
            model: The RWKV model instance
            state_dict: The state dictionary containing the tuned state tensors
        
        Returns:
            initial_state: Combined NumPy array ready to be used with model.eval
        """
        if not state_path:
            return None
        import numpy as np
        import torch
        
        n_layer = 8  # As mentioned in your description
        n_embd = 512  # Based on your state tensor dimensions
        
        # Initialize components for each layer
        all_states = []
        
        # Load the state dictionary using torch.load and convert to numpy
        state_dict = torch.load(state_path, map_location="cpu")
        
        for layer_idx in range(n_layer):
            # 1. Create zero array for attention token shift
            att_token_shift = np.zeros((1, n_embd), dtype=np.float32)
            
            # 2. Create zero array for FFN token shift
            ffn_token_shift = np.zeros((1, n_embd), dtype=np.float32)
            
            # 3. Get the pre-tuned WKV state for this layer
            state_key = f"blocks.{layer_idx}.att.time_state"
            if state_key in state_dict:
                wkv_state = state_dict[state_key].numpy()  # Convert to numpy
                # Extract dimensions and reshape
                head_size = wkv_state.shape[1]
                wkv_state_reshaped = wkv_state.reshape(head_size, n_embd)
            else:
                # If key not found, create a default zero array
                print(f"Warning: {state_key} not found in state dict")
                wkv_state_reshaped = np.zeros((n_embd, n_embd), dtype=np.float32)
            
            # Concatenate the three components for this layer
            layer_state = np.concatenate([
                att_token_shift.flatten(),
                ffn_token_shift.flatten(),
                wkv_state_reshaped.flatten()
            ])
            
            all_states.append(layer_state)
        
        # Concatenate all layer states
        initial_state = np.concatenate(all_states)
        return initial_state
    
    @property
    def device(self) -> torch.device:
        """Returns the device this model is on."""
        return self._device
    
    def to(self, device):
        """Overrides the to() method to track the device."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        # Also move the dummy parameter
        self.dummy = self.dummy.to(device)
        return self
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the model (minimal implementation for generation only).
        """
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise ValueError("Batched generation is not yet supported")
        
        # Convert to CPU for processing
        input_sequence = input_ids[0].cpu().tolist()
        
        if past_key_values is None:
            state = self.initialize_with_tuned_state(self.config.state_path)
            logits, self.current_state = self.model.eval_sequence_in_chunks(
                input_sequence, state, state, None, use_numpy=True
            )
        else:
            # Only process the last token with state
            last_token = input_sequence[-1]
            state, logits_from_past = past_key_values

            logits, self.current_state = self.model.eval(
                last_token, state, state, logits_from_past, use_numpy=True
            )

        # Convert numpy logits to tensor on the device
        logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self._device)
        
        # Return a CausalLMOutputWithPast object instead of a dictionary
        return CausalLMOutputWithPast(
            logits=logits_tensor,
            past_key_values=(self.current_state, logits)
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor, 
        past_key_values=None, 
        attention_mask=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare the input for generation.
        """
        # If we have past, we only need the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder the cache for beam search.
        """
        return past_key_values
    
    def __del__(self):
        """Cleanup resources when the model is deleted."""
        if hasattr(self, 'model'):
            self.model.free()


def create_cpp_model(model_path: str, gpu_layers: int = 99) -> CppModelForCausalLM:
    """
    Create and initialize the C++ model.
    """
    import os
    config = CppModelConfig(model_path=model_path, gpu_layers=gpu_layers, state_path=os.getenv("STATE_PATH"))
    model = CppModelForCausalLM(config)
    return model