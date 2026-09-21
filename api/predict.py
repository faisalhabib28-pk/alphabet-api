from http.server import BaseHTTPRequestHandler
import json
import os
import numpy as np
import string

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


def predict_next_letter(letter):
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


class handler(BaseHTTPRequestHandler):

    def _send(self, status, body):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, x-api-key')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_OPTIONS(self):
        self._send(200, {})

    def do_POST(self):
        expected_key = os.environ.get('API_SECRET_KEY')
        sent_key = self.headers.get('x-api-key')

        if expected_key and sent_key != expected_key:
            self._send(401, {"error": "Invalid or missing API key"})
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b'{}'
            data = json.loads(raw or b'{}')
        except Exception:
            self._send(400, {"error": "Invalid JSON body"})
            return

        letter = data.get('letter', '')
        result = predict_next_letter(letter)

        if result is None:
            self._send(400, {"error": "Sirf a-z letters allowed hain!"})
            return

        self._send(200, {"input": letter, "prediction": result})
