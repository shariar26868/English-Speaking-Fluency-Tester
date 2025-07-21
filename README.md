# IELTS Speaking Fluency Tester
<img width="817" height="427" alt="Screenshot 2025-07-22 043133" src="https://github.com/user-attachments/assets/5aa0213b-9f0b-4213-9bbe-0f67cf3206e2" />

## System Architecture

<img width="1161" height="666" alt="Screenshot 2025-07-22 055848" src="https://github.com/user-attachments/assets/07708dbf-6b91-4117-beec-2e36ef68767c" />

## Project structure
```text
speakingFluencer/
│
├── backend/
│   ├── main.py               # FastAPI app with language parameter
│   ├── evaluate.py           # Multilingual speech analysis
│   ├── feedback.py           # Language-specific feedback
│   ├── utils.py              # Helper functions (e.g., language detection)
│
├── frontend/
│   └── app.py                # Streamlit UI with language selector
│
├── data/
│   ├── history.json          # Stores test results with language metadata
│   └── recordings/           # Audio files (.wav)
│
├── requirements.txt
└── README.md
```
The system follows a client-server architecture with the following components:

### 1. Frontend (Streamlit)
- **File**: `frontend/app.py`
- **Purpose**: Web-based user interface using Streamlit.
- **Functionality**:
  - Record live audio from users.
  - Save recordings to `data/recordings/`.
  - Send audio to backend for evaluation.
  - Display evaluation results (scores, transcript, feedback) and a progress chart.
- **Dependencies**: `streamlit`, `streamlit-audiorecorder`, `pandas`, `altair`.

### 2. Backend (FastAPI)
- **File**: `backend/main.py`
- **Purpose**: Handles API requests and coordinates processing.
- **Functionality**:
  - Receives audio files and processes them.
  - Calls evaluation and feedback modules.
  - Saves results to `data/history.json`.
- **Dependencies**: `fastapi`, `uvicorn`, `pydub`.

### 3. Evaluation Module
- **File**: `backend/evaluate.py`
- **Purpose**: Analyzes audio and calculates scores.
- **Functionality**:
  - Uses Whisper for transcription.
  - Computes fluency (pauses, fillers), grammar (errors, coherence), vocabulary (diversity), and pronunciation (speech rate) scores.
- **Dependencies**: `whisper`, `language_tool_python`, `transformers`.
<img width="797" height="373" alt="Screenshot 2025-07-22 043202" src="https://github.com/user-attachments/assets/6c4ccc95-0ee6-4cad-898f-7d5033634b09" />

### 4. Feedback Module
- **File**: `backend/feedback.py`
- **Purpose**: Generates detailed feedback.
- **Functionality**:
  - Analyzes pauses, filler words, grammar errors, vocabulary repetition, and speech rate.
  - Provides specific suggestions using `language_tool_python` and DistilBERT.
- **Dependencies**: `language_tool_python`, `transformers`.
<img width="780" height="801" alt="Screenshot 2025-07-22 043240" src="https://github.com/user-attachments/assets/56463ffc-1a32-4b44-a529-f141ded46521" />
<img width="780" height="801" alt="Screenshot 2025-07-22 043256" src="https://github.com/user-attachments/assets/17c81a1c-8b21-42b5-8540-083adcb73213" />

### 5. Data Storage
- **Directory**: `data/`
- **Files**:
  - `recordings/`: Stores audio files (e.g., `live_2025-07-22_033900.wav`).
  - `history.json`: Logs all evaluation results with timestamps.
- **Purpose**: Persists user data for progress tracking.

### 6. Utilities
- **File**: `backend/utils.py`
- **Purpose**: Configuration and model loading functions.
- **Functionality**: Language-specific settings and loads the DistilBERT model.

## Workflow

**Flow**:  
User → Frontend (record) → Backend (process) → Evaluation + Feedback → Data Storage → Frontend (display)

### 1. Recording and Submission
- User clicks record on frontend, speaks, and stops the recording.
- Audio saved as `.wav` in `data/recordings/` with a timestamp.
- Sent to backend via `POST /evaluate`.

### 2. Processing
- **Transcription**: Whisper transcribes audio.
- **Fluency**: Measures pause durations and filler words.
- **Grammar**: Uses LanguageTool and DistilBERT for errors and coherence.
- **Vocabulary**: Lexical diversity = unique words / total words.
- **Pronunciation**: Assesses speech rate (words per second).

### 3. Result Delivery
- Backend returns a JSON with scores, transcript, feedback.
- Frontend displays the results and updates the progress chart.
<img width="828" height="522" alt="Screenshot 2025-07-22 043311" src="https://github.com/user-attachments/assets/b4ee2241-9530-4a35-8304-995d078c2d68" />

## Scoring Formulas

### Fluency
- `round((pause_score + filler_score) / 2)`
- Pauses > 2s and filler words each reduce score.
- Range: 1–9

### Grammar
- `round((base_score + coherence_score * 9) / 2)`
- Based on grammar errors and DistilBERT coherence.
- Range: 1–9

### Vocabulary
- `min(9, max(1, lexical_diversity * 10))`
- Diversity = unique_words / total_words
- Range: 1–9

### Pronunciation
- `min(9, max(1, 5 + (words_per_second - target_wps) * 2))`
- Target WPS: 2.0
- Range: 1–9

### Overall Band Score
- `round((fluency + grammar + vocabulary + pronunciation) / 4, 1)`
- Range: 1–9

## Models Used

### Whisper
- Task: Audio transcription
- Output: Transcript + word-level timestamps

### DistilBERT
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Task: Sentence clarity (positivity = coherence)
- Output: Score (0–1 scaled)

### LanguageTool
- Language: en-US
- Task: Grammar, spelling, punctuation checks
- Output: Error matches with suggestions

## Setup Instructions

### Prerequisites
- Python 3.7+
- Git (optional)

### Installation
```bash
git clone https://github.com/yourusername/ielts-speaking-fluency-tester.git
cd ielts-speaking-fluency-tester
```
### How to run
- python -m venv venv
- source venv/Scripts/activate
- pip install -r requirements.txt
- cd backend uvicorn main:app --reload
- cd frontend streamlit run app.py
