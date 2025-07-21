
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from pydub import AudioSegment
import evaluate  # Import the entire module
from feedback import generate_feedback
import json
from datetime import datetime

app = FastAPI()

DATA_DIR = "../data"
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

os.makedirs(RECORDINGS_DIR, exist_ok=True)

@app.post("/evaluate")
async def evaluate_audio(file: UploadFile = File(...), language: str = Form(...)):
    # Validate language
    # supported_languages = ["en", "zh", "hi", "es", "fr", "ar", "bn", "ru", "pt", "ur"]
    supported_languages = ["en"]
    if language not in supported_languages:
        return {"error": "Unsupported language"}

    # Save audio file
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(RECORDINGS_DIR, f"test_{datetime.now().strftime('%Y-%m-%d_%H%M')}.wav")
    
    # Convert to wav if needed
    audio = AudioSegment.from_file(file.file)
    audio.export(audio_path, format="wav")
    
    # Evaluate speech
    transcription, timestamps = evaluate.evaluate_speech(audio_path, language)
    
    # Calculate scores
    fluency_score = evaluate.fluency_score(timestamps, language)
    grammar_score = evaluate.grammar_score(transcription, language)
    vocab_score = evaluate.vocab_score(transcription)
    pronunciation_score = evaluate.pronunciation_score(timestamps, language)
    overall_score = (fluency_score + grammar_score + vocab_score + pronunciation_score) / 4
    
    # Generate feedback
    feedback = generate_feedback(transcription, timestamps, language)
    
    # Save to history
    result = {
        "timestamp": datetime.now().isoformat(),
        "language": language,
        "band_score": round(overall_score, 1),
        "fluency": fluency_score,
        "grammar": grammar_score,
        "vocabulary": vocab_score,
        "pronunciation": pronunciation_score,
        "transcript": transcription,
        "feedback": feedback,
        "audio_path": audio_path
    }
    
    # Update history.json
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    
    history.append(result)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    return result
