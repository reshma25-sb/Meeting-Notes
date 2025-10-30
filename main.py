# app/main.py
import os
import tempfile
import json
from pathlib import Path
from typing import Optional, Any, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI  # ✅ modern import
import uvicorn

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env file")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Meeting Notes & Action Item Extractor")


def call_whisper_transcribe(filepath: str, language: Optional[str] = None) -> str:
    """Transcribe audio using Whisper (new API syntax)."""
    try:
        with open(filepath, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Whisper transcription error: {e}")


def call_llm_structuring(transcript: str) -> Dict[str, Any]:
    """Send transcript to GPT model for structured note extraction."""
    system_prompt = (
        "You are a meeting assistant. Given a meeting transcript, produce a JSON object "
        "with the fields: summary, attendees (list), decisions (list), action_items (list of objects), "
        "agenda_items (list). Each action_item should be {description, assignee (or null), due_date (or null), confidence (0-1)}. "
        "Be concise and conservative; do not invent facts."
    )

    user_prompt = f"Transcript:\n\n{transcript}\n\nReturn only valid JSON with the structure described."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to gpt-4o if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800,
        )
        assistant_text = response.choices[0].message.content
        return json.loads(assistant_text)
    except Exception as e:
        raise RuntimeError(f"LLM structuring failed: {e}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: Optional[str] = None):
    """Upload meeting audio -> get structured notes."""
    suffix = Path(file.filename).suffix or ".wav"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {e}")

    try:
        transcript = call_whisper_transcribe(tmp_path, language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    if not transcript:
        raise HTTPException(status_code=500, detail="Empty transcript returned")

    try:
        structured = call_llm_structuring(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structuring failed: {e}")

    return JSONResponse({
        "transcript": transcript,
        "structured_notes": structured
    })


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
