import argparse 
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from typing import List
from symusic import Score
import miditok
import os

# hyperparams
max_tokens = 1000
temperature = 1.0
top_p = 0.95
repetition_penalty = 1.0
top_k = 40

parser = argparse.ArgumentParser(description='Generate completions from RWKV model based on a prompt')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
parser.add_argument('n_samples', type=int, help='Number of samples to generate')

args = parser.parse_args()

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

midi = Score("./prompt.mid")
config = miditok.TokenizerConfig(
    pitch_range=(0,127),
    use_velocities=False,
    encode_ids_split="no",
    use_pitchdrum_tokens=False,
)
tok = miditok.REMI(config)
break_id = 4
bars_total = 17

os.makedirs('outputs', exist_ok=True)

mc = tok.encode(midi)
emptyPickupMeasure = False
if mc[0].ids[0] == break_id and mc[0].ids[1] == break_id:
    mc[0].ids = mc[0].ids[1:]
    emptyPickupMeasure = True
prompt_tokens: List[int] = mc[0].ids
if prompt_tokens and prompt_tokens[-1] != break_id:
    prompt_tokens.append(break_id)

init_logits, init_state = model.eval_sequence_in_chunks(prompt_tokens, None, None, None, use_numpy=True)

for run in range(args.n_samples):
    build_list_of_tokens = prompt_tokens.copy()
    logits, state = init_logits.copy(), init_state.copy()

    nine_count = sum(1 for token in build_list_of_tokens if token == break_id)

    for i in range(max_tokens):
        if i == 0 or build_list_of_tokens[-1] == break_id:
            logit_bias = {break_id: -100.0}
        else:
            logit_bias = None
        token: int = sampling.sample_logits(logits, temperature=temperature, top_p=top_p, logit_bias=logit_bias, repetition_penalty=repetition_penalty, top_k=top_k, prev_tokens=set(build_list_of_tokens))
        if token >= tok.len:
            token = tok.len - 1

        if token == break_id:
            nine_count += 1
        if nine_count == bars_total:
            break

        build_list_of_tokens.append(token)

        logits, state = model.eval(token, state, state, logits, use_numpy=True)

    if emptyPickupMeasure:
        build_list_of_tokens = [break_id] + build_list_of_tokens
    score = tok.decode([build_list_of_tokens])
    outpath = f'outputs/sample_{run+1:02}.mid'
    score.dump_midi(outpath)