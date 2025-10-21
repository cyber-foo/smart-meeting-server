import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from openai import OpenAI

app = Flask(__name__)
# 必要に応じて許可元を絞ってください（例: origins=["https://your-app.example"]）
CORS(app)

# 環境変数 OPENAI_API_KEY を利用（Render の Environment に設定済みの前提）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
      # 言語ヒントを渡したい場合
      curl -s -X POST https://<your-app>.onrender.com/transcribe \
        -F file=@sample.m4a -F lang=ja
    """
    f = request.files.get("file")
    if not f:
        return (
            jsonify(
                {"error": "file is required (multipart/form-data, key name: 'file')"}
            ),
            400,
        )

    # 任意: 言語ヒント（未指定でもOK）
    lang = request.form.get("lang")  # 例: "ja"

    try:
        # Flask の FileStorage は .stream でバイナリが取れます
        # openai-python は file にバイナリIOを渡せばOK
        filename = secure_filename(f.filename or "audio.m4a")
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f.stream,
            # language=lang,  # 言語を固定したい時はコメント解除
            # response_format="json",  # 既定でOK
        )
        return jsonify({"text": result.text, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/chat")
def chat():
    """
    2つの入力どちらでもOK:
      1) { "prompt": "この文を丁寧語にしてください。よろしく！" }
      2) {
           "messages": [
             {"role":"system","content":"You are a helpful assistant."},
             {"role":"user","content":"自己紹介して"}
           ]
         }

    curl 例:
      curl -s -X POST https://<your-app>.onrender.com/chat \
        -H "Content-Type: application/json" \
        -d '{"prompt":"この文を丁寧語にしてください。よろしく！"}'
    """
    data = request.get_json(silent=True) or {}

    # 柔軟に受け付け
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
@app.route("/test-db")
def test_db():
    try:
        from sqlalchemy import create_engine, text
        import os

        # 環境変数からDB URLを取得
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)

        # テストクエリを実行
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 'Database connected successfully!'"))
            message = result.scalar()

        return {"status": "success", "message": message}

    except Exception as e:
        return {"status": "error", "message": str(e)}
