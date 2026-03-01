from deepface import DeepFace
import numpy as np
from config import THRESHOLD


def get_embedding(frame):
    """Extract face embedding from a frame. Returns None if no face found."""
    try:
        result = DeepFace.represent(
            frame,
            model_name="Facenet",
            enforce_detection=False
        )
        return result[0]["embedding"]
    except Exception:
        return None


def compare_embeddings(emb1, emb2):
    """Return Euclidean distance between two embeddings."""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    return np.linalg.norm(emb1 - emb2)


def is_match(emb1, emb2):
    """Return True if two embeddings are within the recognition threshold."""
    return compare_embeddings(emb1, emb2) < THRESHOLD