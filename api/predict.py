"""
Vercel Serverless Function — /api/predict

Ye endpoint API key check karta hai (header: x-api-key) aur phir
agla-alphabet prediction return karta hai.

API key kahan se aati hai: Vercel dashboard -> Project -> Settings ->
Environment Variables -> naam "API_SECRET_KEY", value aap khud choose karo
(neeche README mein step-by-step tareeqa hai key generate karne ka).
"""
from http.server import BaseHTTPRequestHandler
import json
import os
from infer import predict_next_letter


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
