from collections import Counter
from utils import get_language_config, get_distilbert_model
from transformers import pipeline
import re
import language_tool_python

# Initialize language tool for English
tool = language_tool_python.LanguageTool('en-US')

def generate_feedback(transcription, timestamps, language):
    feedback = []
    lang_config = get_language_config(language)
    classifier = get_distilbert_model()
    
    # Split transcription into sentences for detailed analysis
    sentences = transcription.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Check for pauses
    for i, segment in enumerate(timestamps):
        duration = segment['end'] - segment['start']
        # Flag pauses longer than 2 seconds as excessive
        if duration > 2.0:
            feedback.append({
                "area": "fluency",
                "time": f"{int(segment['start']):02d}:{int(segment['start']%60):02d}–"
                      f"{int(segment['end']):02d}:{int(segment['end']%60):02d}",
                "issue": "Noticeable pause detected",
                "suggestion": (
                    "This pause might disrupt your flow. Try practicing with a timer for 30 seconds "
                    "without stopping, but feel free to use short breaks (under 2 seconds) for natural "
                    "pacing. Record again to see improvement!"
                )
            })
    
    # Check for filler words
    for segment in timestamps:
        if any(filler in segment['text'].lower() for filler in lang_config["fillers"]):
            feedback.append({
                "area": "fluency",
                "time": f"{int(segment['start']):02d}:{int(segment['start']%60):02d}–"
                      f"{int(segment['end']):02d}:{int(segment['end']%60):02d}",
                "issue": "Use of filler words like 'um' or 'uh'",
                "suggestion": (
                    "Fillers can make your speech sound hesitant. Practice speaking slowly and pause "
                    "instead of using 'um' or 'uh'. Try reading a short paragraph aloud without fillers!"
                )
            })
    
    # Grammar and coherence feedback
    for i, sentence in enumerate(sentences):
        if sentence:  # Ensure sentence is not empty
            # DistilBERT coherence check
            result = classifier(sentence, truncation=True, max_length=512)
            coherence_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score']
            
            # LanguageTool grammar check
            matches = tool.check(sentence)
            if matches or coherence_score < 0.7:
                for match in matches:
                    issue = match.message  # Specific error message (e.g., "Possible spelling mistake")
                    offset = match.offset  # Starting position of the error
                    error_length = match.errorLength  # Length of the erroneous text
                    wrong_part = sentence[offset:offset + error_length]  # Extract the erroneous text
                    replacements = match.replacements  # Suggested corrections
                    corrected_sentence = sentence[:offset] + (replacements[0] if replacements else wrong_part) + sentence[offset + error_length:]
                    suggestion = (
                        f"The word or phrase '{wrong_part}' is incorrect. {issue}. "
                        f"Try using '{replacements[0] if replacements else 'a corrected version'}' instead. "
                        f"For example, change '{sentence}' to '{corrected_sentence}' "
                        f"or record again with this fix. Practice this to improve!"
                    )
                    feedback.append({
                        "area": "grammar",
                        "time": f"Sentence {i+1}",
                        "issue": issue,
                        "suggestion": suggestion
                    })
                # Add coherence feedback if no specific errors but low coherence
                if not matches and coherence_score < 0.7:
                    suggestion = "Your sentence might be unclear. Break it into smaller parts or add details, e.g., if you said '{}', try 'I enjoy reading books'."
                    feedback.append({
                        "area": "grammar",
                        "time": f"Sentence {i+1}",
                        "issue": "Sentence may lack clarity or coherence",
                        "suggestion": (
                            f"{suggestion.format(sentence)} Practice recording with simpler structures "
                            "to enhance understanding. Give it another try!"
                        )
                    })
    
    # Vocabulary feedback
    words = transcription.lower().split()
    word_freq = Counter(words)
    common_words = [word for word, count in word_freq.items() if count > len(words) * 0.1]
    if common_words:
        synonym_tips = {
            "i": "Try 'me' or 'myself' in some places, or vary with 'my experience'.",
            "the": "Use 'this' or 'that' to add variety.",
            "is": "Switch with 'seems' or 'appears' for diversity."
        }
        tips = [synonym_tips.get(word, "Explore a thesaurus for alternatives.") for word in common_words]
        feedback.append({
            "area": "vocabulary",
            "time": "Throughout",
            "issue": f"Repetitive use of words: {', '.join(common_words)}",
            "suggestion": (
                f"Repeating words like {', '.join(common_words)} can limit your expressiveness. "
                f"Here are some ideas: {'; '.join(tips)}. Practice using these in your next recording!"
            )
        })
    
    # Pronunciation feedback
    total_duration = sum(t['end'] - t['start'] for t in timestamps)
    word_count = sum(len(t['text'].split()) for t in timestamps)
    words_per_second = word_count / total_duration if total_duration > 0 else 0
    target_wps = lang_config["target_wps"]
    if abs(words_per_second - target_wps) > 0.5:
        adjustment = "slower" if words_per_second > target_wps else "faster"
        feedback.append({
            "area": "pronunciation",
            "time": "Throughout",
            "issue": f"Speech rate of {words_per_second:.1f} words per second differs from the target",
            "suggestion": (
                f"Your speech rate is a bit {adjustment} than the ideal {target_wps:.1f} words per second. "
                f"Practice by reading a text aloud with a metronome or timer, aiming for {target_wps:.1f} words. "
                "Record again to check your progress!"
            )
        })
    
    return feedback