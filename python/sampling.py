import numpy as np
from typing import Dict

# https://stackoverflow.com/a/50425683
def softmax(x: np.ndarray, axis: int):
    x -= x.max(axis=axis, keepdims=True)
    e: np.ndarray = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def sample_logits(
    out,
    temperature: float = 1.0,
    top_p: float = 0.8,
    top_k: int = 0,
    logit_bias: Dict[int, float] = None,
    repetition_penalty: float = 1.0,
    prev_tokens: list = None,
) -> int:
    if hasattr(out, '__module__') and out.__module__ == 'torch':
        out = out.cpu().numpy()

    probs: np.ndarray = softmax(out, axis=-1)

    return sample_probs(
        probs,
        temperature,
        top_p,
        top_k,
        logit_bias,
        repetition_penalty,
        prev_tokens,
    )

def sample_probs(
    probs: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 0.8,
    top_k: int = 0,
    logit_bias: Dict[int, float] = None,
    repetition_penalty: float = 1.0,
    prev_tokens: list = None,
) -> int:
    if not (0.0 <= temperature):
        raise ValueError('temperature')
    if not (0.0 <= top_p <= 1.0):
        raise ValueError('top_p')
    if top_p == 0.0:
        top_p = 1.0

    logits = np.log(probs + 1e-20)  # add epsilon for numerical stability

    # Apply logit bias if provided
    if logit_bias is not None and len(logit_bias) > 0:
        ids, values = zip(*logit_bias.items())
        logits[list(ids)] += values

    # Apply repetition penalty if provided
    if repetition_penalty != 1.0 and prev_tokens is not None and len(prev_tokens) > 0:
        for token in set(prev_tokens):
            if logits[token] > 0:
                logits[token] /= repetition_penalty
            else:
                logits[token] *= repetition_penalty

    # Makes calculation more numerically stable, does not change the result
    logits -= logits.max(axis=-1, keepdims=True)

    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    # Apply top-k
    if top_k > 0 and top_k < len(probs):
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        probs[mask] = 0
        probs = probs / np.sum(probs)

    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, top_p, side="right") + 1
        mask = np.ones_like(probs, dtype=bool)
        mask[sorted_indices[:cutoff]] = False
        probs[mask] = 0
        probs = probs / np.sum(probs)

    # Apply temperature
    if temperature == 0.0:
        return np.argmax(probs).item()
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)

    return np.random.choice(a=len(probs), p=probs)