"""Definition of logits processor used for generation."""

import time

import miditok
import numpy as np
import torch
from miditok import TokSequence
from transformers import LogitsProcessor


class StopLogitsProcessor(LogitsProcessor):
    """

    Custom ``transformers.LogitsProcessor`` implementation.

    Allows stopping generation when enough content to infill bars is generated.

    :param bar_start_token_id: ID of the token indicating the start of a bar.
    :param n_bars_to_infill: number of bars to be infilled in this generation step.
    :param eos_token_id: ID of the EOS (end of sequence) token. If the number
    of bars reaches `max_bars`, the EOS token will be forced to stop generation.

    """

    n_bars_to_infill: int = 0  # This should change at every generation
    # step as we may need to infill a different number of bars at each step
    n_attribute_controls: int = 0  # Number of attribute controls to skip
    # when decoding using BPE
    infill_type: str = None

    def __init__(
        self,
        bar_start_token_id: int,
        eos_token_id: int,
        track_start_token_id: int,
        track_end_token_id: int,
        tokenizer: miditok.MusicTokenizer
    ) -> None:
        self.bar_start_token_id = bar_start_token_id
        self.eos_token_id = eos_token_id
        self.track_start_token_id = track_start_token_id
        self.track_end_token_id = track_end_token_id
        self.tokenizer = tokenizer
        self.total_time = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        To handle proper infilling generation content.

        Assert that the right number of bars are generated
        for infilling and that the generation is stopped when all
        the bars are generated.

        :param input_ids: ids of the input sequence of tokens
        :param scores: pre-softmax sampling probabilities
        :return: output tokens prediction scores
        """
        start_time = time.time()

        generated_tokens = TokSequence(are_ids_encoded=True)

        if self.infill_type == "bar":
            fill_start_idx = np.where(
                input_ids[0].numpy() == self.tokenizer.vocab["FillBar_Start"]
            )[0][0]
        elif self.infill_type == "track":
            fill_start_idx = np.where(
                input_ids[0].numpy() == self.tokenizer.vocab["Infill_Track"]
            )[0][0]

        n_bar_none = 0
        if fill_start_idx + self.n_attribute_controls + 1 < len(input_ids[0]):
            generated_tokens.ids = input_ids[0][
                fill_start_idx + self.n_attribute_controls + 1 :
            ].tolist()
            self.tokenizer.decode_token_ids(generated_tokens)

            n_bar_none = len(
                np.where(
                    np.array(generated_tokens.ids) == self.tokenizer.vocab["Bar_None"]
                )[0]
            )

        penalty = 999999.0

        # If we reach the self.n_bars_to_infill + 1 BarStart token sampled,
        # we have generated enough content
        if n_bar_none > self.n_bars_to_infill:
            scores[:, :] = -penalty  # Penalize all tokens
            # But enforce the sampling of EOS token to stop generation
            scores[:, self.eos_token_id] = penalty

        # Don't sample an EOS token until all bars are generated
        if n_bar_none <= self.n_bars_to_infill:
            scores[:, self.eos_token_id] = -penalty

        end_time = time.time()
        self.total_time += end_time - start_time

        # don't sample track start/end
        scores[:, self.track_start_token_id] = -penalty
        scores[:, self.track_end_token_id] = -penalty
        # mess around with 797 and 665 logits
        scores[:, 797] = -penalty

        scores[:, 4] = -penalty # Infill_Track

        return scores