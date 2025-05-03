import torch
import torch.nn.functional as F
from miditok import MMM, TokSequence
from typing import Optional
from transformers import LogitsProcessorList
import rwkv_cpp_shared_library, rwkv_cpp_model

class CppModelConfig:
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


class CustomGenerator:
    def __init__(self, config: CppModelConfig, tokenizer: MMM):
        # Load the C++ model
        self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        self.model = rwkv_cpp_model.RWKVModel(
            self.library, 
            config.model_path, 
            gpu_layer_count=config.gpu_layers
        )
        self.tokenizer = tokenizer    
        self.current_state = None
        self.state_path = config.state_path

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

        # print(self.tokens_have_bar_none_and_timesig)
        # print(self.tokens_beginning_timesig)
        # print(self.tokens_ending_bar_none)
        # print("--------------------")

        # for token in self.tokens_have_bar_none_and_timesig:
        #     t = TokSequence(ids=[token], are_ids_encoded=True)
        #     tokenizer.decode_token_ids(t)
        #     print(t.tokens)


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
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 100,
        epsilon_cutoff: float = 0.0,
        do_sample: bool = True,
        logits_processor: LogitsProcessorList = None,
        attribute_controls: list = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens with optional sampling and token injection.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens  
            num_beams: Number of beams for beam search (1 = greedy/sampling)
            temperature: Temperature for sampling
            repetition_penalty: Penalty for repeating tokens
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            max_new_tokens: Maximum number of new tokens to generate
            epsilon_cutoff: Probability threshold below which tokens are excluded
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Token ID for padding
            eos_token_id: Token ID indicating end of sequence
            token_injection_rules: Dictionary mapping trigger token IDs to tokens to inject
            
        Returns:
            generated_ids: The generated token IDs
        """
        batch_size = input_ids.shape[0]
        
        if batch_size > 1:
            raise ValueError("Batched generation is not yet supported")
        
        # Process initial input sequence
        input_sequence = input_ids[0].cpu().tolist()
        current_sequence = input_sequence.copy()
        
        # Initialize state with the entire input sequence
        state = self.initialize_with_tuned_state(self.state_path)
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
        
        while tokens_generated < max_new_tokens:
            # Convert logits to next_token_logits format (batch_size, vocab_size)
            next_token_logits = logits_tensor.clone()

            next_token_scores = logits_processor(current_sequence, next_token_logits)
            
            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_scores = next_token_scores / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_token in prev_tokens_set:
                    if prev_token < next_token_scores.size(-1):  # Safety check
                        next_token_scores[0, prev_token] /= repetition_penalty
            
            # Apply epsilon cutoff
            if epsilon_cutoff > 0:
                # Create a mask for tokens below the probability threshold
                probs = F.softmax(next_token_scores, dim=-1)
                next_token_scores[probs < epsilon_cutoff] = -float('inf')
            
            if do_sample:
                # Apply top-k filtering
                if 0 < top_k < next_token_scores.size(-1):
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_scores, top_k, dim=-1, largest=True, sorted=True
                    )
                    
                    # Create a new tensor with -inf everywhere
                    filtered_logits = torch.full_like(next_token_scores, -float('inf'))
                    
                    # Scatter the top-k logits back to the original tensor
                    filtered_logits[0, top_k_indices[0]] = top_k_logits[0]
                    
                    next_token_scores = filtered_logits
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_scores, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
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
            
            # Process the generated token through the model
            logits, current_state = self.model.model.eval(
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
                # TODO: is this a proper stopping criterion?
                if ac_idx > len(attribute_controls):
                    break

                injection_tokens = [tokenizer.vocab[ac] for ac in attribute_controls[ac_idx]]
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
            if eos_token_id is not None and next_token_id == self.tokenizer.vocab["EOS_None"]:
                break
        
        # Return the complete sequence (input + generated)
        generated_tensor = torch.tensor(current_sequence, dtype=torch.long).unsqueeze(0)
        return generated_tensor


# Example usage:
def example_usage():
    """
    Example showing how to use the custom generator with a C++ model.
    """
    # Load the model
    from your_model_file import create_cpp_model  # your import path
    model = create_cpp_model("model_path", gpu_layers=99)
    
    # Define token injection rules
    token_injection_rules = {
        1000: [2000, 3000],  # When token 1000 appears, inject tokens 2000 and 3000
        500: 600,            # When token 500 appears, inject token 600
    }
    
    # Create generator
    generator = CustomGenerator(model)
    
    # Generate with token injection
    input_ids = torch.tensor([[1, 2, 3, 4]])
    outputs = generator.generate(
        input_ids=input_ids,
        num_beams=1,
        temperature=1.0,
        repetition_penalty=1.2,
        top_k=20,
        top_p=0.95,
        max_new_tokens=500,
        epsilon_cutoff=9e-4,
        do_sample=True,
        token_injection_rules=token_injection_rules
    )
    
    print("Generated sequence:", outputs[0].tolist())

if __name__ == "__main__":
    from pathlib import Path
    current_dir = Path("/home/christian/MIDI-RWKV/src/inference")
    TOK_PATH = current_dir.parent / "tokenizer/tokenizer_with_acs.json"
    MODEL_PATH = str(current_dir.parent / "outputs/m2fla/rcpp.bin")
    INPUT_PATH = str(current_dir / "mat/rollinggirlCON.mid")
    # INPUT_PATH = "/home/christian/MIDI-RWKV/RWKV-PEFT/data/disc_1/01_Introducing_The_Beatles__A_Taste_Of_Honey.mid"
    OUTPUT_PATH = str(current_dir / "mat/output.mid")
    OUTWAV_PATH = str(current_dir / "mat/output.wav")
    INWAV_PATH = str(current_dir / "mat/input.wav")
    OUTPR_PATH = str(current_dir / "mat/output.png")
    INPR_PATH = str(current_dir / "mat/input.png")
    
    tokenizer = MMM(params=TOK_PATH)
    config = CppModelConfig(MODEL_PATH, 0, "")
    CustomGenerator(config, tokenizer)