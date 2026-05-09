from deepface import DeepFace
import numpy as np
import time
from config import THRESHOLD_HIGH, THRESHOLD_LOW

# Global tracking to prevent spamming logs
LAST_PERSON = None
LAST_ANNOUNCEMENT_TIME = 0
ANNOUNCE_COOLDOWN = 20 

def get_embedding(frame):
    """Extract FaceNet embedding. Uses opencv backend for speed."""
    try:
        # We use enforce_detection=False to prevent the code from crashing if a face isn't clear
        result = DeepFace.represent(
            frame,
            model_name="Facenet",
            detector_backend="opencv",
            align=True,
            normalization="Facenet",
            enforce_detection=False
        )
        if not result: return None
        emb = result[0]["embedding"]
        return emb.tolist() if hasattr(emb, "tolist") else emb
    except Exception as e:
        print(f"[FACE ERROR] {e}")
        return None

def cosine_similarity(emb1, emb2):
    a, b = np.array(emb1), np.array(emb2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def find_best_match(embedding, known_persons):
    """Compares current face against database with improved threshold logic."""
    best_score = -1
    best_person = None

    for person in known_persons:
        similarities = [cosine_similarity(embedding, emb) for emb in person["embeddings"]]
        if not similarities: continue
        
        avg_score = sum(similarities) / len(similarities)
        if avg_score > best_score:
            best_score = avg_score
            best_person = person

    # 1. Strong Match
    if best_score >= THRESHOLD_HIGH:
        return best_person, best_score
    
    # 2. Uncertain Match (Prevents jumping immediately to Unknown)
    if best_score >= THRESHOLD_LOW:
        # Create a temporary 'Maybe' object to show in UI
        maybe_person = best_person.copy()
        maybe_person["name"] = f"Maybe {best_person['name']}"
        return maybe_person, best_score

    return None, 0.0