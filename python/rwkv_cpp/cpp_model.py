import torch
import torch.nn.functional as F
import numpy as np
from miditok import MMM, TokSequence
from transformers import LogitsProcessorList, GenerationConfig
from . import rwkv_cpp_shared_library, rwkv_cpp_model

class CppModelConfig:
    """Configuration class for the C++ model wrapper."""
    model_type: str = "cpp_model"
    
    def __init__(
        self,
        model_path: str = "",
        state_path: str = "",
        **kwargs
    ):
        self.model_path = model_path
        self.state_path = state_path


class CustomGenerator:
    def __init__(self, config: CppModelConfig, tokenizer: MMM):
        # Load the C++ model
        self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        self.model = rwkv_cpp_model.RWKVModel(
            self.library, 
            config.model_path, 
            gpu_layer_count=0
        )
        self.tokenizer = tokenizer    
        self.current_state = None
        self.state = self.initialize_with_tuned_state(config.state_path)

        self.tokens_ending_bar_none = []
        self.tokens_beginning_timesig = []
        self.tokens_have_bar_none_and_timesig = []
        for i in range(tokenizer.vocab_size):
            t = TokSequence(ids=[i], are_ids_encoded=True)
            tokenizer.decode_token_ids(t)
            if len(t.tokens) == 0:
                continue
            if t.tokens[-1] == "Bar_None":
                self.tokens_ending_bar_none.append(i)
            if "TimeSig" in t.tokens[0]:
                self.tokens_beginning_timesig.append(i)
            if "Bar_None" in t.tokens and any("TimeSig" in x for x in t.tokens):
                self.tokens_have_bar_none_and_timesig.append(i)

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
        
        n_layer = self.model.n_layer
        n_embd = self.model.n_embed
        
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
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig = None,
        logits_processor: LogitsProcessorList = None,
        attribute_controls: list = None,
    ) -> torch.LongTensor:
        batch_size = input_ids.shape[0]
        
        if batch_size > 1:
            raise ValueError("Batched generation is not yet supported")
        
        # Process initial input sequence
        input_sequence = input_ids[0].cpu().tolist()
        current_sequence = input_sequence.copy()
        
        # Initialize state with the entire input sequence
        state = self.state.copy() if self.state is not None else None
        # attribnute controls are preinjected for bar infilling
        logits, current_state = self.model.eval_sequence_in_chunks(
            input_sequence, state, state, None, use_numpy=True
        )
        
        # Track previous tokens for repetition penalty
        prev_tokens_set = set(input_sequence)
        
        # Convert logits to tensor on the device
        logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
        
        # Keep track of tokens that were actually generated (not injected)
        tokens_generated = 0
        did_last_token_end_in_bar_none = False
        ac_idx = 1

        while tokens_generated < generation_config.max_new_tokens:
            # Convert logits to next_token_logits format (batch_size, vocab_size)
            next_token_logits = logits_tensor.clone()

            next_token_scores = logits_processor(current_sequence, next_token_logits) if logits_processor else next_token_logits
            
            # Apply temperature scaling
            if generation_config.temperature > 0 and generation_config.temperature != 1.0:
                next_token_scores = next_token_scores / generation_config.temperature
            
            # Apply repetition penalty
            if generation_config.repetition_penalty != 1.0:
                for prev_token in prev_tokens_set:
                    if prev_token < next_token_scores.size(-1):  # Safety check
                        next_token_scores[0, prev_token] /= generation_config.repetition_penalty
            
            # Apply epsilon cutoff
            if generation_config.epsilon_cutoff > 0:
                # Create a mask for tokens below the probability threshold
                probs = F.softmax(next_token_scores, dim=-1)
                next_token_scores[probs < generation_config.epsilon_cutoff] = -float('inf')
            
            if generation_config.do_sample:
                # Apply top-k filtering
                if 0 < generation_config.top_k < next_token_scores.size(-1):
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_scores, generation_config.top_k, dim=-1, largest=True, sorted=True
                    )
                    
                    # Create a new tensor with -inf everywhere
                    filtered_logits = torch.full_like(next_token_scores, -float('inf'))
                    
                    # Scatter the top-k logits back to the original tensor
                    filtered_logits[0, top_k_indices[0]] = top_k_logits[0]
                    
                    next_token_scores = filtered_logits
                
                # Apply top-p (nucleus) filtering
                if generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_scores, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set removed indices to -inf
                    indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
                    next_token_scores[0, indices_to_remove] = -float('inf')
                
                # Convert logits to probabilities and sample
                probs = F.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            
            next_token_id = next_token[0, 0].item()
            if next_token_id == 797:
                next_token_id = 665

            # Process the generated token through the model
            logits, current_state = self.model.eval(
                next_token_id, current_state, current_state, logits, use_numpy=True
            )
            logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)

            # Add to generated tokens and current sequence
            current_sequence.append(next_token_id)
            
            # Update previous tokens set for repetition penalty
            prev_tokens_set.add(next_token_id)

            #### ----------------------- TOKEN INJECTION ----------------------- ####

            did_last_token_end_in_bar_none = next_token_id in self.tokens_ending_bar_none

            if attribute_controls is not None and len(attribute_controls) > 1 and ((next_token_id in self.tokens_beginning_timesig and did_last_token_end_in_bar_none) or next_token_id in self.tokens_have_bar_none_and_timesig):
                if ac_idx >= len(attribute_controls):
                    break

                injection_tokens = [self.tokenizer.vocab[ac] for ac in attribute_controls[ac_idx]]
                ac_idx += 1

                for injected_token_id in injection_tokens:
                    logits, current_state = self.model.eval(
                        injected_token_id, current_state, current_state, logits, use_numpy=True
                    )
                    
                    # Update tracking
                    current_sequence.append(injected_token_id)
                
                # Update logits_tensor for the next iteration with the final injected token's logits
                logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
            
            tokens_generated += 1
            
            # Check if we've generated an EOS token and can stop early
            if any(next_token_id == self.tokenizer.vocab[x] for x in ["FillBar_End", "Track_End", "EOS_None"]):
                break
        
        # Return the complete sequence (input + generated)
        generated_tensor = torch.tensor(current_sequence, dtype=torch.long).unsqueeze(0)
        return generated_tensor
