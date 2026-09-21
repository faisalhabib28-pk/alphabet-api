import numpy as np
import string
import os

alphabets = string.ascii_lowercase
le_map = {ch: idx for idx, ch in enumerate(alphabets)}
idx_map = {idx: ch for ch, idx in le_map.items()}

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')
_weights = None


def _load_weights():
    global _weights
    if _weights is None:
        _weights = np.load(_WEIGHTS_PATH)
    return _weights


def _relu(x):
    return np.maximum(0, x)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_next_letter(letter: str):
    letter = (letter or "").strip().lower()
    if len(letter) == 0:
        return None
    last_char = letter[-1]
    if last_char not in le_map:
        return None

    w = _load_weights()
    x = np.zeros(26)
    x[le_map[last_char]] = 1.0

    h1 = _relu(x @ w['W1'] + w['b1'])
    h2 = _relu(h1 @ w['W2'] + w['b2'])
    out = _softmax(h2 @ w['W3'] + w['b3'])

    return idx_map[int(np.argmax(out))]
