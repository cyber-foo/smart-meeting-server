import os
import io
import json
from typing import Optional   # ← 追加
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ← あなたのキーをサーバ側にのみ保存
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")  # 任意（あれば）

app = FastAPI(title="SmartMeeting Transcription API")

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    lang: Optional[str] = None,                 # ← 修正
    x_app_token: Optional[str] = Header(default=None)  # ← 修正
):
    # 例: 共有トークンでの簡易保護（必要なければ削除OK）
    APP_TOKEN = os.getenv("APP_TOKEN", "")
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="server api key missing")

    # Whisper REST に転送
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT

    form = {
        "model": (None, "whisper-1"),
    }
    if lang:
        form["language"] = (None, lang)

    try:
        content = await audio.read()
        files = {
            "file": (audio.filename or "audio.wav", content, audio.content_type or "application/octet-stream")
        }
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, headers=headers, files=files, data=form)
        if resp.status_code // 100 == 2:
            data = resp.json()
            # 期待: {"text": "..."}
            return JSONResponse({"text": data.get("text", "")})
        else:
            raise HTTPException(status_code=resp.status_code, detail="transcription failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
