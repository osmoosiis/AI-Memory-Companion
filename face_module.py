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

    Returns (person_dict, score, confident) where:
      - confident=True  → score >= THRESHOLD_HIGH → safe to announce
      - confident=False → score in [THRESHOLD_LOW, THRESHOLD_HIGH) → show "Maybe" in UI only
      - (None, 0.0, False) → no match found

    The 'name' field is NEVER prefixed with 'Maybe' — callers decide display logic.
    """
    best_score  = -1.0
    best_person = None

    for person in known_persons:
        embeddings = person.get("embeddings", [])
        if not embeddings:
            continue

        similarities = [cosine_similarity(embedding, emb) for emb in embeddings]

        # Top-3 average instead of all-embedding average.
        # With augmented datasets, a straight average is dragged down by
        # weaker augmented variants (flips, brightness shifts), causing false Unknowns.
        top_k      = min(3, len(similarities))
        top_scores = sorted(similarities, reverse=True)[:top_k]
        score      = sum(top_scores) / top_k

        if score > best_score:
            best_score  = score
            best_person = person

    if best_score >= THRESHOLD_HIGH:
        return best_person, best_score, True   # confident match

    if best_score >= THRESHOLD_LOW:
        return best_person, best_score, False  # uncertain — show in UI, don't announce

    return None, 0.0, False  # no match