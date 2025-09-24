import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()  # 環境変数 OPENAI_API_KEY を自動で利用

@app.get("/")
def root():
    return "SmartMeeting Server is running!"

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/transcribe")
def transcribe():
    """
    curl 例:
      curl -s -X POST https://<your-app>.onrender.com/transcribe \
        -F file=@sample.m4a
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "file is required (multipart/form-data, key=name 'file')"}), 400

    # OpenAI Whisper で文字起こし
    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",       # Whisper API
            file=f,                  # そのままファイルオブジェクトを渡せます
            # language="ja",         # 日本語を明示したい時はコメント解除
            # response_format="json" # 既定でOK
        )
        return jsonify({"text": result.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/chat")
def chat():
    """
    curl 例:
      curl -s -X POST https://<your-app>.onrender.com/chat \
        -H "Content-Type: application/json" \
        -d '{"prompt":"この文を丁寧語にしてください。よろしく！"}'
    """
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "JSON body with 'prompt' is required"}), 400

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # 軽量・安価で十分賢い
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
