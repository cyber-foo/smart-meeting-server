import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Android からのクロスオリジン呼び出しを許可
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def root():
    return "SmartMeeting Server is running!"

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

# ---------- ChatGPT: テキスト→返答 ----------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        system_prompt = data.get("system", "You are a helpful assistant.")
        model = data.get("model", "gpt-4o-mini")

        if not user_message:
            return jsonify({"error": "No 'message' provided"}), 400

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        reply = resp.choices[0].message.content
        return jsonify({"reply": reply, "model": model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Whisper: 音声→文字起こし ----------
# multipart/form-data で 'audio' フィールドに音声ファイルを付けて送ってください
@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file field 'audio' found"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Whisper (英語/多言語)
        # 'language' を指定したい場合は form-data で渡せます（例: ja, en）
        language = request.form.get("language")
        kwargs = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language

        result = client.audio.transcriptions.create(**kwargs)
        # SDK 1.x は result.text に文字起こし結果
        return jsonify({"text": result.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Render で受けるため 0.0.0.0 固定
    app.run(host="0.0.0.0", port=port)
