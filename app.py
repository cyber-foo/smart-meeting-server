import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Android から直接叩けるように
import requests

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
# 必要に応じて Origin を絞ってください（開発中は "*" でOK）
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def root():
    return "SmartMeeting Server is running!"

@app.route("/healthz", methods=["GET"])
def healthz():
    # OPENAI_API_KEY が設定されているかも簡易チェック
    return jsonify({
        "status": "ok",
        "has_api_key": bool(OPENAI_API_KEY)
    })

@app.route("/chat", methods=["POST"])
def chat():
    """
    リクエスト:  { "message": "こんにちは" }
    レスポンス: { "reply": "..." }
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

        data = request.get_json(silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        if not user_msg:
            return jsonify({"error": "message is required"}), 400

        # --- OpenAI Chat Completions（REST） ---
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",  # 適宜変更可
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for a meeting app."},
                {"role": "user",   "content": user_msg}
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return jsonify({"error": "OpenAI API error", "detail": r.text}), 502

        j = r.json()
        reply = j["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": "server error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
