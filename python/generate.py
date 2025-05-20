"""Tests for MMM inference."""

import random
import time
import traceback
from pathlib import Path
import os
from copy import deepcopy
import numpy as np
import torch.cuda as cuda
from miditok import MMM
from miditok.utils import get_bars_ticks, get_beats_ticks, get_score_ticks_per_beat
from miditok.attribute_controls import BarNoteDensity, BarNoteDuration, BarOnsetPolyphony
from symusic import Score
from transformers import AutoModelForCausalLM, GenerationConfig
from inference import InferenceConfig, generate
from rwkv_cpp.cpp_model import CustomGenerator, CppModelConfig

NUM_BEAMS = 1
TEMPERATURE_SAMPLING = float(os.getenv("TEMPERATURE_SAMPLING", 1.0))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.0))
TOP_K = int(os.getenv("TOP_K", 20))
TOP_P = float(os.getenv("TOP_P", 0.95))
EPSILON_CUTOFF = 9e-4 # None
ETA_CUTOFF = None
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
MAX_LENGTH = 99999

HERE = Path(__file__).parent
subdir = "pop909_test" if os.getenv("pop909") else "gigamidi_test"
MIDI_PATHS = list((HERE / "test_midis" / subdir).glob("**/*.mid"))


def test_generate(tokenizer: MMM,
                  models: list[tuple[str, AutoModelForCausalLM]],
                  gen_config: GenerationConfig,
                  input_midi_path: str | Path):

    MIDI_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Get number of tracks and number of bars of the MIDI track
    score = Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    num_tracks = len(tokens)

    # Select random track index to infill
    track_idx = random.randint(0, num_tracks-1) if not os.getenv("pop909") else 0

    if DRUM_GENERATION:
        programs = [tokens[idx].tokens[1] for idx in range(num_tracks)]
        if "Program_-1" not in programs:
            return False
        track_idx = programs.index("Program_-1")
    else:
        # If not generating drums, skip until we sample
        # a non drum track index
        attempts = 0
        while tokens[track_idx].tokens[1] == "Program_-1":
            track_idx = np.random.randint(0, num_tracks)
            attempts += 1
            if attempts > 40:
                return False

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(bars_ticks)


    if END_INFILLING and not os.getenv("partial_end", False):
        bar_idx_infill_start = num_bars - NUM_BARS_TO_INFILL
    else:
        one_end = CONTEXT_SIZE // 4
        other_end = (num_bars - CONTEXT_SIZE - NUM_BARS_TO_INFILL - 1) // 4
        bar_idx_infill_start = (random.randint(one_end, other_end) if other_end > one_end else random.randint(other_end, one_end)) * 4

    # Compute stuff to discard infillings when we have no context!
    bar_left_context_start = bars_ticks[
        bar_idx_infill_start - CONTEXT_SIZE
    ]
    bar_infilling_start = bars_ticks[bar_idx_infill_start]

    if not END_INFILLING or os.getenv("partial_end", False):
        bar_infilling_end = bars_ticks[bar_idx_infill_start + NUM_BARS_TO_INFILL]

        try:
            bar_right_context_end = bars_ticks[
                bar_idx_infill_start + NUM_BARS_TO_INFILL + CONTEXT_SIZE
            ]
        except IndexError:
            bar_right_context_end = bars_ticks[-1]

    times = np.array([event.time for event in tokens[track_idx].events])
    types = np.array([event.type_ for event in tokens[track_idx].events])
    tokens_left_context_idxs = np.nonzero((times >= bar_left_context_start) &
                                          (times <= bar_infilling_start))[0]
    tokens_left_context_types = set(types[tokens_left_context_idxs])
    if END_INFILLING and not os.getenv("partial_end", False):
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
    
    
    # extract attribute controls
    density_control = BarNoteDensity(18)
    duration_control = BarNoteDuration()
    polyphony_control = BarOnsetPolyphony(1, 6)
    ticks_bars = get_bars_ticks(score, only_notes_onsets=True)
    ticks_beats = get_beats_ticks(score, only_notes_onsets=True)
    try:
        density_controls = density_control.compute(score.tracks[track_idx], score.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start-1, bar_idx_infill_start + NUM_BARS_TO_INFILL)))[1:]
        duration_controls = duration_control.compute(score.tracks[track_idx], score.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start, bar_idx_infill_start + NUM_BARS_TO_INFILL)))
        polyphony_controls = polyphony_control.compute(score.tracks[track_idx], score.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start, bar_idx_infill_start + NUM_BARS_TO_INFILL)))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing attribute controls: {e}")

    if len(density_controls) == 0 or len(duration_controls) == 0 or len(polyphony_controls) == 0:
        print(
            f"[WARNING::test_generate] Infilling region"
            f"{bar_idx_infill_start} - "
            f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            "has no attribute controls!"
        )
        return False
    
    try:
        acl = []
        for i in range(NUM_BARS_TO_INFILL):
            this_bar_acl = [polyphony_controls[2*i], polyphony_controls[2*i+1], density_controls[i], duration_controls[5*i], duration_controls[5*i+1], duration_controls[5*i+2], duration_controls[5*i+3], duration_controls[5*i+4]] if not DRUM_GENERATION else [density_controls[i]]
            partial_acl = [f"{x.type_}_{x.value}" for x in this_bar_acl]
            acl.append(partial_acl)
    except IndexError:
        # if it doesn't get all NUM_BARS_TO_INFILL bars
        return False

    inference_config = InferenceConfig(
            CONTEXT_SIZE,
            {
                track_idx: [
                    (
                        bar_idx_infill_start,
                        bar_idx_infill_start + NUM_BARS_TO_INFILL,
                        acl,
                        "bar"
                    )
                ],
            },
            [],
        )
    
    outputs = {}

    try:
        for name, model in models:
            start_time = time.time()

            outputs[name] = output = generate(
                model,
                tokenizer,
                inference_config,
                input_midi_path,
                {"generation_config": gen_config},
                input_tokens=deepcopy(tokens)
            )

            end_time = time.time()
            ticks_bars = get_bars_ticks(output, only_notes_onsets=True)
            ticks_beats = get_beats_ticks(output, only_notes_onsets=True)
            after_density_controls = density_control.compute(output.tracks[track_idx], output.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start-1, bar_idx_infill_start + NUM_BARS_TO_INFILL)))[1:]
            after_duration_controls = duration_control.compute(output.tracks[track_idx], output.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start, bar_idx_infill_start + NUM_BARS_TO_INFILL)))
            after_polyphony_controls = polyphony_control.compute(output.tracks[track_idx], output.ticks_per_quarter, ticks_bars, ticks_beats, list(range(bar_idx_infill_start, bar_idx_infill_start + NUM_BARS_TO_INFILL)))

            def prec(x):
                if x == "18+":
                    return 19
                return int(x)
            density_deltas = [abs(prec(x.value) - prec(y.value)) for x, y in zip(density_controls, after_density_controls)]
            duration_deltas = [abs(int(x.value) - int(y.value)) for x, y in zip(duration_controls, after_duration_controls)]
            polyphony_deltas = [abs(int(x.value) - int(y.value)) for x, y in zip(polyphony_controls, after_polyphony_controls)]

            if os.getenv("evaluate_acs", False):
                with open(f"acs_{name}_nbi{NUM_BARS_TO_INFILL}.txt", "a") as f:
                    f.write(str({
                        "density_deltas": density_deltas,
                        "duration_deltas": duration_deltas,
                        "polyphony_deltas": polyphony_deltas
                    }))
                    f.write("\n")

    except Exception as e: # noqa: BLE001
        print(f"An unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()  # full stack trace

        return False
    
    for name in outputs.keys():
        outputs[name].dump_midi(
            MIDI_OUTPUT_FOLDER / f"{input_midi_path.stem}_track{track_idx}_"
            f"infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+NUM_BARS_TO_INFILL}"
            f"_context_{CONTEXT_SIZE}"
            f"_generationtime_{round(end_time - start_time, 3)}_{name}.mid"
        )
    
    return True

MODEL_PATH = None

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
    parser.add_argument("-m", "--mistral",
                        type=lambda x: x.lower() in ["true", "1", "yes"],
                        required=True,
                        help="Boolean flag for Mistral (True/False)")

    # Parse arguments
    args = parser.parse_args()

    NUM_BARS_TO_INFILL = args.num_bars_infilling
    CONTEXT_SIZE = args.context
    DRUM_GENERATION = args.drums
    END_INFILLING = args.end_infilling
    MISTRAL = args.mistral
    proj_root = os.getenv("PROJECT_ROOT", "~/MIDI-RWKV")

    MODEL_PATH = os.getenv("MODEL_PATH") if not MISTRAL else f"{proj_root}/rwkv.cpp/python/MIDI_Mistral"

    additional_flags = "_"
    if DRUM_GENERATION:
        additional_flags += "drums"
    if END_INFILLING:
        additional_flags += "endinfilling"
    if os.getenv("pop909"):
        additional_flags += "_pop909"

    # MIDI-Mistral uses tokenizer.json which doesn't support ACs
    tokenizer = MMM(params=f"{proj_root}/train/tokenizer/tokenizer_with_acs.json" if not MISTRAL else f"{proj_root}/train/tokenizer/tokenizer.json")

    # switch these in and out for different models
    # first string is the name of the model as it will appear in the filename
    models = [
        ("base", CustomGenerator(CppModelConfig(MODEL_PATH, ""), tokenizer)),
        # ("lora32", CustomGenerator(CppModelConfig(f"{proj_root}/rwkv.cpp/python/rwkv_cpp/merged_32.bin", ""), tokenizer)),
        # ("lora4", CustomGenerator(CppModelConfig(f"{proj_root}/rwkv.cpp/python/rwkv_cpp/merged.bin", ""), tokenizer)),
        # ("state1", CustomGenerator(CppModelConfig(MODEL_PATH, f"{proj_root}/RWKV-PEFT/peft_model/lr0.05_100x16/rwkv-16.pth"), tokenizer)),
        # ("state2", CustomGenerator(CppModelConfig(MODEL_PATH, f"{proj_root}/RWKV-PEFT/peft_model/lr0.05_100x16_v2/rwkv-16.pth"), tokenizer)),
        # ("state3", CustomGenerator(CppModelConfig(MODEL_PATH, f"{proj_root}/RWKV-PEFT/peft_model/rwkv-16.pth"), tokenizer)),
        # ("mistral", AutoModelForCausalLM.from_pretrained(f"{proj_root}/rwkv.cpp/python/MIDI_Mistral", device_map="auto")),
    ]

    random.seed(42)

    # used for appendix
    # gen_params = [
    #     (1.2, 1.2, 20, 0.95),
    #     (0.8, 1.2, 20, 0.95),
    #     (1.0, 1.0, 20, 0.95),
    #     (1.0, 1.4, 20, 0.95),
    #     (1.0, 1.2, 15, 0.95),
    #     (1.0, 1.2, 30, 0.95),
    #     (1.0, 1.2, 20, 0.98),
    #     (1.0, 1.2, 20, 0.90),
    # ]

    gen_params = [(1.0, 1.2, 20, 0.95)]

    for TEMPERATURE_SAMPLING, REPETITION_PENALTY, TOP_K, TOP_P in gen_params:
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
            do_sample=True,
        )

        i = 0
        while i < args.num_generations:
            midi_file = random.choice(MIDI_PATHS)
            try:
                if test_generate(tokenizer, models, gen_config, midi_file):
                    i += 1
                    print("------ successful generation ------")
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass