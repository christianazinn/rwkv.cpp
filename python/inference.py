"""Inference method for the MMM model."""

from __future__ import annotations

import re
import os
import time
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import LogitsProcessorList

from logits_processor import StopLogitsProcessor

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from config import InferenceConfig


def generate(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score_or_path: Score | Path | str,
    generate_kwargs: Mapping | None = None,
    input_tokens: TokSequence | list[TokSequence] = None
) -> Score:
    """
    Use the model to generate new music content.

    The method allows to infill specific bars or generate new tracks.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param score_or_path: ``symusic.Score`` or path of the music file to infill.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    logits_processor = StopLogitsProcessor(
        tokenizer.vocab["Bar_None"], tokenizer.vocab["FillBar_End"], tokenizer.vocab["Track_Start"], tokenizer.vocab["Track_End"], tokenizer
    )

    input_tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    # Infill bars
    if inference_config.infilling:
        score = generate_infilling(
            model, tokenizer, inference_config, logits_processor,
            generate_kwargs, deepcopy(input_tokens)
        )

    # Generate new tracks
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            score = generate_new_track(model, tokenizer, track, generate_kwargs)

    return score


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Generate a new track of a given Score.

    The new track will be added to the score.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track
        Attribute Controls.
    :param score: symusic.Score
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # In this case, the prompt is a toksequence containing all the tracks
    input_seq = tokenizer.encode(score)

    # Add <TRACK_START> and <PROGRAM> tokens
    input_seq.ids.append(tokenizer.vocab["Track_Start"])
    input_seq.tokens.append("Track_Start")
    input_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    input_seq.tokens.append(f"Program_{track[0]}")

    # Add attribute control tokens
    for control in track[1]:
        input_seq.ids.append(tokenizer.vocab[control])
        input_seq.tokens.append(control)

    output_ids = model.generate(LongTensor([input_seq.ids]), **generate_kwargs)
    output_seq = TokSequence(ids=output_ids[0].tolist(), are_ids_encoded=True)

    # Remove attribute controls from the sequence
    output_seq = (
        output_seq[: len(input_seq)] + output_seq[len(input_seq) + len(track[1]) :]
    )

    # Decode BPE ids before getting the associated tokens
    tokenizer.decode_token_ids(output_seq)
    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    # It is expected to have a <TRACK_END> token at the end of the sequence.
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation failed: the model failed to predict a <TRACK_END> token",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    return tokenizer._tokens_to_score(output_seq)


def generate_infilling(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    logits_processor: StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
    input_tokens: TokSequence | list[TokSequence]  = None
) -> Score:
    """
    Generate a new portion of a ``symusic.Score``.

    The portion to infill will be generated with the model and added to the score
    inplace for the selected tracks. Notes originally present in the portion to
    infill will be removed.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param score: ``symusic.Score`` to generate a new track from.
    :param inference_config: InferenceConfig
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop
        generation when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab[
            "FillBar_End"
        ]

    tracks_to_infill = inference_config.bars_to_generate.keys()

    start_time = time.time()
    # input_tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    """
    Just for debugging purposes
    print_input_tokens = tokenizer.encode(score)

    tokenizer.decode_token_ids(print_input_tokens)
    with open("original_tokens.txt", "w") as file:
        bar_n = 0
        track_n = 0
        for token in print_input_tokens.tokens:
            if token == "Track_End":
                bar_n = 0
                track_n += 1
            if token == "Bar_None":
                file.write(f"TrackNumber:{track_n} BarNumber:{bar_n} " + token + "\n")
                bar_n += 1
            else:
                file.write(token + "\n")
    """

    end_time = time.time()
    print(
        "[INFO::generate_infilling] Time spent for converting score to tokens: ",
        end_time - start_time,
    )

    for track_to_infill in tracks_to_infill:
        infill_bars(
            model,
            tokenizer,
            track_to_infill,
            inference_config,
            input_tokens,
            logits_processor,
            generate_kwargs,
        )

    # Here we use the base tokenizer because output_tokens is a list of TokSequences

    start_time = time.time()
    result = tokenizer.base_tokenizer._tokens_to_score(input_tokens)
    end_time = time.time()
    print(
        "[INFO::generate_infilling] Time spent for converting tokens to score: ",
        end_time - start_time,
    )
    return result


def infill_bars(
    model: object,
    tokenizer: MMM,
    track_idx: int,
    inference_config: InferenceConfig,
    tokens: list[TokSequence],
    logits_processor: StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> None:
    """
    Infill bars for the ''track_idx'' track.

    The tokens are replaced inplace.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to
        generate.
    :param tokens: TokSequence of the track to be infilled
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation
        when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # For each set of bars to infill in the track, we generate new content
    # (We may have, in the same track, non-adjacent sequences of bars. For
    # each sequence, we do a generation step).
    for subset_bars_to_infill in inference_config.bars_to_generate[track_idx]:
        # token_start_idx and token_end_idx are the indices of start
        # and end of infilling, when the toksequence is NOT BPE encoded
        start_time = time.time()

        input_seq, token_start_idx, token_end_idx = _adapt_prompt_for_infilling(
            tokenizer, track_idx, tokens, subset_bars_to_infill,
            inference_config.context_length
        )

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spent for creating input sequence: ",
            end_time - start_time,
        )

        logits_processor.n_bars_to_infill = (
            subset_bars_to_infill[1] - subset_bars_to_infill[0]
        )
        logits_processor.n_attribute_controls = len(subset_bars_to_infill[2])
        logits_processor.infill_type = subset_bars_to_infill[3]
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        start_time = time.time()

        new_generated_tokens = TokSequence(are_ids_encoded=True)
        offset = 0

        for i in range(logits_processor.n_bars_to_infill):
            print(input_seq.ids)
            output_ids = model.generate(
                LongTensor([input_seq.ids]),
                logits_processor=logit_processor_list,
                **generate_kwargs,
            )[0].numpy()

            end_time = time.time()
            generation_time = end_time - start_time
            print("[INFO::infill_bars] Time spent for generation: ", generation_time)
            print("Time spent in logits processor ", logits_processor.total_time)

            start_time = time.time()

            if subset_bars_to_infill[3] == "bar":
                fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]
            elif subset_bars_to_infill[3] == "track":
                fill_start_idx = np.where(output_ids == tokenizer.vocab["Infill_Track"])[0][0]

            # Here we isolate the generated tokens doing some filtering. In particular,
            # the model may generate some tokens before the first Bar_None token
            generated_tokens = TokSequence(are_ids_encoded=True)
            print("output ids")
            print(output_ids.tolist())
            # print(new_generated_tokens.ids)
            generated_tokens.ids = output_ids[
                fill_start_idx + len(subset_bars_to_infill[2]) + offset + 2 : -1
            ].tolist()
            print(generated_tokens.ids)
            # decode_token_ids doesn't support numpy arrays for ids list
            # print(generated_tokens.ids)
            tokenizer.decode_token_ids(generated_tokens)
            if len(generated_tokens.ids) > 0 and generated_tokens.ids[0] != tokenizer.vocab["Bar_None"]:
                generated_tokens.ids.insert(0, tokenizer.vocab["Bar_None"])
                generated_tokens.ids.insert(1, tokenizer.vocab["TimeSig_4/4"])
            bar_none_token_idxs = np.where(
                np.array(generated_tokens.ids) == tokenizer.vocab["Bar_None"]
            )[0]
            # print(bar_none_token_idxs)
            print(generated_tokens.tokens)
            try:
                generated_tokens.ids = generated_tokens.ids[
                    bar_none_token_idxs[0] : bar_none_token_idxs[1]
                    # bar_none_token_idxs[0] : bar_none_token_idxs[logits_processor.n_bars_to_infill]
                ]
                print("gentokid")
                new_generated_tokens.ids += generated_tokens.ids
                print(generated_tokens.ids)
                generated_tokens.ids.append(tokenizer.vocab["Bar_None"])
                generated_tokens.ids.append(tokenizer.vocab["TimeSig_4/4"])
                attribute_controls = subset_bars_to_infill[2]
                for control in attribute_controls:
                    generated_tokens.ids.append(tokenizer.vocab[control])
                tokenizer.encode_token_ids(generated_tokens)
                print("----------------------asdf")
                print(input_seq.ids[-10:])
                print(generated_tokens.ids[:10])
                print("-------------")
                input_seq.ids += generated_tokens.ids
                # new_generated_tokens.ids += generated_tokens.ids
                offset += len(generated_tokens.ids)
                # TODO: this is a stupid blunt tool
                input_seq.ids = [x if x != 797 else 665 for x in input_seq.ids]
            except Exception:
                pass
                # formerly: pass
                # TODO: does this improve?
                # raise ValueError(
                #     f"[WARNING::infill_bars] Ignoring infilling of bars "
                #     f"{subset_bars_to_infill[0]} - "
                #     f"{subset_bars_to_infill[1]} on track {track_idx}"
                #     " because the model failed to generate a sequence of the right length"
                # )
        print("---------------done--------------------")
        print(new_generated_tokens.ids)
        tokenizer.decode_token_ids(new_generated_tokens)
        print(new_generated_tokens.ids)

        tokens[track_idx].ids[token_start_idx:token_end_idx] = new_generated_tokens.ids
        tokens[track_idx].tokens = tokenizer._ids_to_tokens(tokens[track_idx].ids)

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spend for reconstructing the sequence: ",
            end_time - start_time,
        )


def _adapt_prompt_for_infilling(
    tokenizer: MMM,
    track_idx: int,
    tokens: list[TokSequence],
    subset_bars_to_infill: tuple[int, int, list[str]],
    context_length: int
) -> TokSequence:
    """
    Construct the prompt for bar infilling.

    Constructs the prompt to be used as model's input. The sequence should have the
    "BAR_FILL" format:
    ``<TRACK_START>...<TRACK_END>...<TRACKS_START>...<INFILL_BAR>...<INFILL_BAR>...
    <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>``
    We have as many <FILL_IN> tokens as the number of bars we want to infill.

    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param tokens: TokSequence of the track to be infilled
    :param subset_bars_to_infill: contains the indexes of the first and last bar to
        infill, plus a list of attribute controls
    """
    toksequence_to_infill: TokSequence = TokSequence(are_ids_encoded=False)

    # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
    # at the right place
    tokenizer.decode_token_ids(tokens)

    start_bar_idx = subset_bars_to_infill[0]
    end_bar_idx = subset_bars_to_infill[1]

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(tokens[track_idx]._ticks_bars)

    #### Infilling tokens ####
    times = np.array([event.time for event in tokens[track_idx].events])

    token_idx_start = np.nonzero(times >= bars_ticks[start_bar_idx])[0][0]

    # In this case, infilling is done until the end of the track
    if end_bar_idx >= num_bars:
        token_idx_end = len(tokens[track_idx]) - 1
    else:
        token_idx_end = np.nonzero(times >= bars_ticks[end_bar_idx])[0][0]

    if subset_bars_to_infill[3] == "bar":
        ##### Infilling Context#####
        
        # If the number of bars available as context in the left
        # is less than context length
        if start_bar_idx - context_length < 0:
            context_token_start_idx = 2
        else:
            context_token_start_idx = np.nonzero(
                times >= bars_ticks[start_bar_idx - context_length]
            )[0][0]
        
        # Take right context
        if end_bar_idx < num_bars:
            # If the number of bars available as context in the right
            # is less than context length
            if end_bar_idx + context_length > num_bars - 1:
                context_token_end_idx = len(tokens[track_idx]) - 1
            else:
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[end_bar_idx + context_length]
                )[0][0]
        # Only left context (aka END INFILLING)
        else:
            context_token_end_idx = -1
        
        seq_before = (
            tokens[track_idx][:2]
            + tokens[track_idx][context_token_start_idx:token_idx_start]
        )
        for _ in range(end_bar_idx - start_bar_idx):
            seq_before.ids.append(tokenizer.vocab["Infill_Bar"])
            seq_before.tokens.append("Infill_Bar")
        seq_after = tokens[track_idx][token_idx_end:context_token_end_idx]
        toksequence_to_infill += seq_before + seq_after
        toksequence_to_infill.ids.append(tokenizer.vocab["Track_End"])
        toksequence_to_infill.tokens.append("Track_End")
        

    output_toksequence = TokSequence(are_ids_encoded=False)

    ###### Context for other tracks #####
    for i in range(len(tokens)):
        if subset_bars_to_infill[3] == "bar":
            if i == track_idx:
                output_toksequence += toksequence_to_infill
                continue
    
            #with open("model_prompt_tokens.txt", "w") as file:
            #    for token in output_toksequence.tokens:
            #        file.write(token + "\n")
    
            times = np.array([event.time for event in tokens[i].events])
            if start_bar_idx - context_length < 0:            
                context_token_start_idx = 0
            else:
                try:
                    context_token_start_idx = np.nonzero(
                        times >= bars_ticks[start_bar_idx - context_length]
                    )[0][0]
                except IndexError:
                    context_token_start_idx = 0
            if end_bar_idx + context_length >= num_bars - 1:
                context_token_end_idx = len(tokens[i]) - 1
            else:
                # TODO: number of bars of some tracks computed through
                #   miditok is not always right, meaning that the list
                #   of tokens after the context may be out of the allowed
                #   range
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[end_bar_idx + context_length]
                )[0]
                # In that case, we just take the last token as the end of context
                if len(context_token_end_idx) == 0:
                    context_token_end_idx = len(tokens[i]) - 1
                else:
                    context_token_end_idx = context_token_end_idx[0]
    
            # Add the section to the context only if it is not empty
            sliced_tokens = tokens[i][context_token_start_idx:context_token_end_idx]
    
            pattern = r"Pitch"
            if any(re.match(pattern, token) for token in sliced_tokens.tokens):
                output_toksequence += (
                    tokens[i][:2]
                    + sliced_tokens
                    + tokens[i][-1:]
                )
        elif subset_bars_to_infill[3] == "track":
            if i == track_idx:
                infill_program_id = tokens[i][1]
                infill_program_token = tokens[i].tokens[1]
                continue

            times = np.array([event.time for event in tokens[i].events])

            #### Infilling tokens ####

            token_idx_start_ = np.nonzero(times >= bars_ticks[start_bar_idx])[0][0]

            # In this case, infilling is done until the end of the track
            if end_bar_idx >= num_bars:
                token_idx_end_ = len(tokens[i]) - 1
            else:
                token_idx_end_ = np.nonzero(times >= bars_ticks[end_bar_idx])[0][0]

            # Add the section to the context only if it is not empty
            sliced_tokens = tokens[i][token_idx_start_:token_idx_end_]

            pattern = r"Pitch"
            if any(re.match(pattern, token) for token in sliced_tokens.tokens):
                output_toksequence += (
                        tokens[i][:2]
                        + sliced_tokens
                        + tokens[i][-1:]
                )
    
    if subset_bars_to_infill[3] == "bar":
        output_toksequence.ids.append(tokenizer.vocab["FillBar_Start"])
        output_toksequence.tokens.append("FillBar_Start")
        output_toksequence.ids.append(tokenizer.vocab["Bar_None"])
        output_toksequence.tokens.append("Bar_None")
        output_toksequence.ids.append(tokenizer.vocab["TimeSig_4/4"])
        output_toksequence.tokens.append("TimeSig_4/4")
    elif subset_bars_to_infill[3] == "track":
        output_toksequence.ids.append(tokenizer.vocab["Infill_Track"])
        output_toksequence.tokens.append("Infill_Track")
        output_toksequence.ids.append(infill_program_id)
        output_toksequence.tokens.append(infill_program_token)
            
    # TODO: not for mistral
    attribute_controls = subset_bars_to_infill[2]
    for control in attribute_controls:
        output_toksequence.ids.append(tokenizer.vocab[control])
        output_toksequence.tokens.append(control)

    # Encode into BPE tokens
    tokenizer.encode_token_ids(output_toksequence)

    print(output_toksequence.ids)
    print(output_toksequence.tokens)
    print("-------------------------------")

    #Just for debugging purposes

    # with open("model_prompt_tokens.txt", "w") as file:
    #     bar_n = 0
    #     track_n = 0
    #     for token in output_toksequence.tokens:
    #         if token == "Track_End":
    #             bar_n = 0
    #             track_n += 1
    #         if token == "Bar_None":
    #             file.write(f"TrackNumber:{track_n} BarNumber:{bar_n} " + token + "\n")
    #             bar_n += 1
    #         else:
    #             file.write(token + "\n")

    return output_toksequence, token_idx_start, token_idx_end

def _adapt_prompt_for_track_infilling(
    tokenizer: MMM,
    track_idx: int,
    tokens: list[TokSequence],
    subset_bars_to_infill: tuple[int, int, list[str]],
) -> TokSequence:
    pass


if __name__ == "__main__":
    from transformers import GenerationConfig
    from symusic import Synthesizer, dump_wav
    from pathlib import Path
    from rwkv_cpp.cpp_model import create_cpp_model
    trk = 0
    acl = ['ACBarOnsetPolyphonyMin_1', 'ACBarOnsetPolyphonyMax_3', 'ACBarNoteDensity_8', 'ACBarNoteDurationWhole_0', 'ACBarNoteDurationHalf_0', 'ACBarNoteDurationQuarter_1', 'ACBarNoteDurationEight_1', 'ACBarNoteDurationSixteenth_1']
    INFERENCE_CONFIG = InferenceConfig(
        bars_to_generate={
            # "ACTrackOnsetPolyphonyMin_1", "ACTrackOnsetPolyphonyMax_6", "ACBarOnsetPolyphonyMin_1", "ACBarPitchClass_11", "ACTrackNoteDensityMin_8", "ACBarNoteDensity_6", "ACBarNoteDurationEight_1", "ACTrackRepetition_1.00"
            trk: [(14, 16, acl, "bar")],
        },
        new_tracks=[
            # (0, ["ACBarPitchClass_3", "ACTrackNoteDensityMax_4", "ACTrackRepetition_0.89"]),
        ],
        context_length=4
    )

    gen_config = GenerationConfig(
            num_beams=1,
            temperature=1.0,
            repetition_penalty=1.2,
            top_k=20,
            top_p=0.95,
            max_new_tokens=300,
            epsilon_cutoff=9e-4,
            # ALWAYS test once with do_sample=False
            do_sample=True,
        )
    
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
    model = create_cpp_model(MODEL_PATH)

    print("Generating...")

    output_scores = generate(
            model,
            tokenizer,
            INFERENCE_CONFIG,
            INPUT_PATH,
            {"generation_config": gen_config},
        )
    
    print("Done generating. Dumping MIDI...")

    # Create MIDI output
    output_scores.dump_midi(OUTPUT_PATH)
    print("Dumped MIDI, synthesizing...")

    # # synth
    synth = Synthesizer()
    outscore = Score(OUTPUT_PATH)
    inscore = Score(INPUT_PATH)
    outwav = synth.render(outscore, stereo=True)
    inwav = synth.render(inscore, stereo=True)
    dump_wav(OUTWAV_PATH, outwav, sample_rate=44100, use_int16=True)
    dump_wav(INWAV_PATH, inwav, sample_rate=44100, use_int16=True)

    # print("Synthesized, plotting piano rolls...")

    # matplotlib
    from matplotlib import pyplot as plt
    intrack = inscore.resample(tpq=6, min_dur=1).tracks[trk].pianoroll(modes=["onset", "frame"], pitch_range=[0, 128], encode_velocity=False)
    outtrack = outscore.resample(tpq=6, min_dur=1).tracks[trk].pianoroll(modes=["onset", "frame"], pitch_range=[0, 128], encode_velocity=False)

    a = 200
    b = 500
    intrack_truncated = [intrack[0][:, a:b], intrack[1][:, a:b]]
    outtrack_truncated = [outtrack[0][:, a:b], outtrack[1][:, a:b]]
    plt.imshow(intrack_truncated[0] + intrack_truncated[1], aspect="auto", origin="lower")
    plt.savefig(INPR_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    plt.imshow(outtrack_truncated[0] + outtrack_truncated[1], aspect="auto", origin="lower")
    plt.savefig(OUTPR_PATH, dpi=300, bbox_inches="tight")
    

