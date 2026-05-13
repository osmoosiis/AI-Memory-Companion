"""
face_module.py — FaceNet embedding extraction and matching.

No changes required; preserved verbatim from original.
"""

from deepface import DeepFace
import numpy as np
from config import THRESHOLD_HIGH, THRESHOLD_LOW


def get_embedding(frame):
    """Extract FaceNet embedding using OpenCV backend for speed."""
    try:
        result = DeepFace.represent(
            frame,
            model_name="Facenet",
            detector_backend="opencv",
            align=True,
            normalization="Facenet",
            enforce_detection=False,
        )
        if not result:
            return None
        emb = result[0]["embedding"]
        return emb.tolist() if hasattr(emb, "tolist") else emb
    except Exception as e:
        print(f"[FACE ERROR] {e}")
        return None


def cosine_similarity(emb1, emb2) -> float:
    a, b  = np.array(emb1), np.array(emb2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def find_best_match(embedding, known_persons):
    """
    Compare embedding against all stored persons.

    Returns (person_dict, score) for best match, or (None, 0.0) if no match
    exceeds THRESHOLD_LOW.

    Scores in [THRESHOLD_LOW, THRESHOLD_HIGH) return a 'Maybe <name>' person
    to show uncertainty in the UI without triggering a full announcement.
    """
    best_score  = -1.0
    best_person = None

    for person in known_persons:
        embeddings = person.get("embeddings", [])
        if not embeddings:
            continue

        similarities = [cosine_similarity(embedding, emb) for emb in embeddings]
        avg_score    = sum(similarities) / len(similarities)

        if avg_score > best_score:
            best_score  = avg_score
            best_person = person

    if best_score >= THRESHOLD_HIGH:
        return best_person, best_score

    if best_score >= THRESHOLD_LOW:
        maybe = best_person.copy()
        maybe["name"] = f"Maybe {best_person['name']}"
        return maybe, best_score

    return None, 0.0