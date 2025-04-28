"""Tests for MMM inference."""

import random
import time
import traceback
from pathlib import Path
import os

import numpy as np
import torch.cuda as cuda
from miditok import MMM
from symusic import Score
from transformers import AutoModelForCausalLM, GenerationConfig
from inference import InferenceConfig, generate
from rwkv_cpp.cpp_model import create_cpp_model

NUM_BEAMS = 1
TEMPERATURE_SAMPLING = float(os.getenv("TEMPERATURE_SAMPLING", 1.0))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.0))
TOP_K = int(os.getenv("TOP_K", 20))
TOP_P = float(os.getenv("TOP_P", 0.95))
EPSILON_CUTOFF = None
ETA_CUTOFF = None
MAX_NEW_TOKENS = 300
MAX_LENGTH = 99999

HERE = Path(__file__).parent
subdir = "beatles_test_set" if os.getenv("beatles") else "test_midis"
MIDI_PATHS = list((HERE / subdir).glob("**/*.mid"))


def test_generate(tokenizer: MMM,
                  model: AutoModelForCausalLM,
                  gen_config: GenerationConfig,
                  input_midi_path: str | Path):

    MIDI_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Get number of tracks and number of bars of the MIDI track
    score = Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    num_tracks = len(tokens)

    # Select random track index to infill
    track_idx = random.randint(0, num_tracks-1)

    if DRUM_GENERATION:
        programs = [tokens[idx].tokens[1] for idx in range(num_tracks)]
        if "Program_-1" not in programs:
            return False
        track_idx = programs.index("Program_-1")
    else:
        # If not generating drums, skip until we sample
        # a non drum track index
        while tokens[track_idx].tokens[1] == "Program_-1":
            track_idx = random.randint(0, num_tracks-1)
            continue

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(bars_ticks)


    if END_INFILLING:
        bar_idx_infill_start = num_bars - NUM_BARS_TO_INFILL
    else:
        bar_idx_infill_start = random.randint(
            CONTEXT_SIZE // 4, (num_bars - CONTEXT_SIZE - NUM_BARS_TO_INFILL - 1) // 4
        ) * 4

    # Compute stuff to discard infillings when we have no context!
    bar_left_context_start = bars_ticks[
        bar_idx_infill_start - CONTEXT_SIZE
    ]
    bar_infilling_start = bars_ticks[bar_idx_infill_start]

    if not END_INFILLING:
        bar_infilling_end = bars_ticks[bar_idx_infill_start + NUM_BARS_TO_INFILL]

        bar_right_context_end = bars_ticks[
            bar_idx_infill_start + NUM_BARS_TO_INFILL + CONTEXT_SIZE
        ]

    times = np.array([event.time for event in tokens[track_idx].events])
    types = np.array([event.type_ for event in tokens[track_idx].events])
    tokens_left_context_idxs = np.nonzero((times >= bar_left_context_start) &
                                          (times <= bar_infilling_start))[0]
    tokens_left_context_types = set(types[tokens_left_context_idxs])
    if END_INFILLING:
        tokens_infilling = np.nonzero(times >= bar_infilling_start)[0]
    else:
        tokens_infilling = np.nonzero((times >= bar_infilling_start) &
                                      (times <= bar_infilling_end))[0]
    if not END_INFILLING:
        tokens_right_context_idxs = np.nonzero((times >= bar_infilling_end) &
                                               (times <= bar_right_context_end))[0]
        tokens_right_context_types = set(types[tokens_right_context_idxs])

    pitch_token = "Pitch"
    if DRUM_GENERATION:
        pitch_token = "PitchDrum"

    if END_INFILLING:
        if pitch_token not in tokens_left_context_types:
            print(
                f"[WARNING::test_generate] Ignoring infilling of bars "
                f"{bar_idx_infill_start} - "
                f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
                " because we have no context around the infilling region"
            )
            return False
    elif (pitch_token not in tokens_left_context_types or
          pitch_token not in tokens_right_context_types):
        print(
            f"[WARNING::test_generate] Ignoring infilling of bars "
            f"{bar_idx_infill_start} - "
            f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            f"on file {input_midi_path}"
            " because we have no context around the infilling region"
        )
        return False

    if len(tokens_infilling) == 0:
        print(
            f"[WARNING::test_generate] Infilling region"
            f"{bar_idx_infill_start} - "
            f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            "has no notes!"
        )
        return False

    inference_config = InferenceConfig(
            CONTEXT_SIZE,
            {
                track_idx: [
                    (
                        bar_idx_infill_start,
                        bar_idx_infill_start + NUM_BARS_TO_INFILL,
                        [],
                        "bar"
                    )
                ],
            },
            [],
        )

    try:
        start_time = time.time()

        _ = generate(
            model,
            tokenizer,
            inference_config,
            input_midi_path,
            {"generation_config": gen_config},
            input_tokens=tokens
        )

        end_time = time.time()
    except Exception as e: # noqa: BLE001
        print(f"An unexpected error occurred during generation: {e}")
        traceback.print_exc()  # full stack trace
        return False
    
    _.dump_midi(
            MIDI_OUTPUT_FOLDER / f"{input_midi_path.stem}_track{track_idx}_"
            f"infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+NUM_BARS_TO_INFILL}"
            f"_context_{CONTEXT_SIZE}"
            f"_generationtime_{round(end_time - start_time, 5)}.mid"
        )

    return True

MODEL_PATH = os.getenv("MODEL_PATH")#  if not MISTRAL else "/home/christian/MIDI-RWKV/rwkv.cpp/python/MISTRAL_123000"
# "/home/christian/MIDI-RWKV/src/outputs/m2fla/rcpp.bin"

CONTEXT_SIZE = None

NUM_BARS_TO_INFILL = None

DRUM_GENERATION = False

END_INFILLING = False

DEBUG = False

MIDI_OUTPUT_FOLDER = None

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate MIDI sequences "
                                                 "with specified parameters.")
    parser.add_argument("-nbi", "--num_bars_infilling", type=int, required=True,
                        help="Number of bars for infilling")
    parser.add_argument("-c", "--context", type=int, required=True,
                        help="Context length")
    parser.add_argument("-g", "--num_generations",
                        type=int, required=True, help="Number of generations")
    parser.add_argument("-d", "--drums",
                        type=lambda x: x.lower() in ["true", "1", "yes"],
                        required=True,
                        help="Boolean flag for drums (True/False)")
    parser.add_argument("-e", "--end_infilling",
                        type=lambda x: x.lower() in ["true", "1", "yes"],
                        required=True, help="Boolean flag for infilling end")
    # parser.add_argument("-m", "--mistral",
    #                     type=lambda x: x.lower() in ["true", "1", "yes"],
    #                     required=True,
    #                     help="Boolean flag for Mistral (True/False)")

    # Parse arguments
    args = parser.parse_args()

    NUM_BARS_TO_INFILL = args.num_bars_infilling
    CONTEXT_SIZE = args.context
    DRUM_GENERATION = args.drums
    END_INFILLING = args.end_infilling

    additional_flags = "_"
    if DRUM_GENERATION:
        additional_flags += "drums"
    if END_INFILLING:
        additional_flags += "endinfilling"
    if os.getenv("beatles"):
        additional_flags += "_beatles"

    MIDI_OUTPUT_FOLDER = (Path(__file__).parent
                          / "output"
                          /"TEST_TRACK_INFILLING" 
                          f"temp{TEMPERATURE_SAMPLING}"
                            f"_rep{REPETITION_PENALTY}"
                            f"_topK{TOP_K}_topP{TOP_P}"
                            f"num_bars_infill{NUM_BARS_TO_INFILL}_context{CONTEXT_SIZE}{additional_flags}")

    if os.path.exists(MIDI_OUTPUT_FOLDER):
        import shutil
        shutil.rmtree(MIDI_OUTPUT_FOLDER)

    tokenizer = MMM(params="/home/christian/MIDI-RWKV/src/tokenizer/tokenizer_with_acs.json")#  if not MISTRAL else "/home/christian/MIDI-RWKV/src/tokenizer/tokenizer.json")

    model = create_cpp_model(MODEL_PATH) # if not MISTRAL else AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    gen_config = GenerationConfig(
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE_SAMPLING,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        epsilon_cutoff=EPSILON_CUTOFF,
        eta_cutoff=ETA_CUTOFF,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        do_sample = True
    )

    i = 0
    while i < args.num_generations:
        midi_file = random.choice(MIDI_PATHS)
        try:
            if test_generate(tokenizer, model, gen_config, midi_file):
                i += 1
        except:
            pass