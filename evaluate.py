
import whisper
import language_tool_python
from collections import Counter
import numpy as np
from utils import get_language_config, get_distilbert_model
from transformers import pipeline

# Load Whisper base model
model = whisper.load_model("base")

# Initialize language tools for supported languages
grammar_tools = {
    "en": language_tool_python.LanguageTool('en-US'),
    "es": language_tool_python.LanguageTool('es'),
    "fr": language_tool_python.LanguageTool('fr'),
    "ru": language_tool_python.LanguageTool('ru'),
    "pt": language_tool_python.LanguageTool('pt-PT')
}

def fluency_score(timestamps, language):
    lang_config = get_language_config(language)
    pauses = [t['end'] - t['start'] for t in timestamps]
    filler_words = sum(1 for t in timestamps if any(filler in t['text'].lower() for filler in lang_config["fillers"]))
    pause_score = min(9, max(1, 9 - len([p for p in pauses if p > 1.0]) * 0.5))
    filler_score = min(9, max(1, 9 - filler_words * 0.5))
    return round((pause_score + filler_score) / 2)

def grammar_score(text, language):
    lang_config = get_language_config(language)
    if language in grammar_tools:
        matches = grammar_tools[language].check(text)
        error_count = len(matches)
        base_score = max(1, 9 - error_count * 0.5)
    else:
        base_score = 5  # Default for unsupported languages

    # Use DistilBERT for coherence scoring
    classifier = get_distilbert_model()
    result = classifier(text, truncation=True, max_length=512)
    coherence_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score']
    return round((base_score + coherence_score * 9) / 2)

def vocab_score(text):
    words = text.lower().split()
    unique_words = len(set(words))
    total_words = len(words)
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    return min(9, max(1, lexical_diversity * 10))

def pronunciation_score(timestamps, language):
    lang_config = get_language_config(language)
    total_duration = sum(t['end'] - t['start'] for t in timestamps)
    word_count = sum(len(t['text'].split()) for t in timestamps)
    words_per_second = word_count / total_duration if total_duration > 0 else 0
    target_wps = lang_config["target_wps"]
    return min(9, max(1, 5 + (words_per_second - target_wps) * 2))

def evaluate_speech(audio_path, language):
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    transcription = result["text"]
    timestamps = result["segments"]
    return transcription, timestamps
