import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.get("/")
def root():
    return "SmartMeeting Server is running!"


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.post("/transcribe")
def transcribe():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "file is required (multipart/form-data, key name: 'file')"}), 400

    lang = request.form.get("lang")

    try:
        filename = secure_filename(f.filename or "audio.m4a")
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f.stream,
        )
        return jsonify({"text": result.text, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    prompt = data.get("prompt")

    if not messages and not prompt:
        return jsonify({"error": "Provide 'messages' or 'prompt' in JSON body."}), 400

    if not messages:
        messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ ここを「app.run()」より上に移動
@app.route("/test-db")
def test_db():
    try:
        from sqlalchemy import create_engine, text
        import os

        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 'Database connected successfully!'"))
            message = result.scalar()

        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ✅ 最後にこれ
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
