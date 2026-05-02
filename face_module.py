from deepface import DeepFace
import numpy as np
from config import THRESHOLD_HIGH, THRESHOLD_LOW


def get_embedding(frame):
    """Fast + consistent embedding"""
    try:
        result = DeepFace.represent(
            frame,
            model_name="Facenet",
            detector_backend="opencv",  # 🔥 faster than retinaface
            align=True,
            normalization="Facenet",
            enforce_detection=False
        )

        emb = result[0]["embedding"]

        if hasattr(emb, "tolist"):
            emb = emb.tolist()

        return emb

    except Exception:
        return None


def cosine_similarity(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0:
        return 0.0

    return float(np.dot(emb1, emb2) / denom)


def find_best_match(embedding, known_persons):
    best_person = None
    best_score = -1.0

    for person in known_persons:
        similarities = []

        for emb in person["embeddings"]:
            try:
                sim = cosine_similarity(embedding, emb)
                similarities.append(sim)
            except:
                continue

        if len(similarities) == 0:
            continue

        avg_score = float(sum(similarities) / len(similarities))

        print(f"[DEBUG] {person['name']} score: {avg_score:.3f}")

        if avg_score > best_score:
            best_score = avg_score
            best_person = person

    print(f"[FINAL BEST] score: {best_score:.3f}")

    # 🔥 HYSTERESIS LOGIC (NO FLICKER)
    if best_score > THRESHOLD_HIGH:
        return best_person
    elif best_score > THRESHOLD_LOW and best_person:
        return {
            "name": f"Maybe {best_person['name']}",
            "relationship": best_person["relationship"],
            "reminder": best_person["reminder"]
        }
    else:
        return None