import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from insightface.app import FaceAnalysis


DATASET_ROOT = Path("CAIDE_DATA")
SIMILARITY_THRESHOLD = 0.10


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_gallery(dataset_root: Path):
    gallery_embeddings = []
    gallery_ids = []
    metadata_map = {}
    photo_map = {}

    for person_dir in sorted(dataset_root.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("P"):
            continue

        mean_path = person_dir / "arcface_mean_vector.json"
        meta_path = person_dir / "metadata.json"
        photo_path = person_dir / "1.jpg"

        if not mean_path.exists() or not meta_path.exists():
            print(f"Skipping {person_dir.name}: missing required files")
            continue

        with open(mean_path, "r", encoding="utf-8") as f:
            mean_data = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        emb = np.asarray(mean_data["mean_embedding"], dtype=np.float32)
        emb = l2_normalize(emb)

        gallery_embeddings.append(emb)
        gallery_ids.append(person_dir.name)
        metadata_map[person_dir.name] = meta_data
        photo_map[person_dir.name] = str(photo_path) if photo_path.exists() else None

    if not gallery_embeddings:
        raise RuntimeError("No gallery embeddings found.")

    gallery_matrix = np.stack(gallery_embeddings, axis=0)
    return gallery_matrix, gallery_ids, metadata_map, photo_map


def top_k_matches(query_embedding, gallery_matrix, gallery_ids, k=3):
    scores = gallery_matrix @ query_embedding
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    for idx in ranked_indices[:k]:
        pid = gallery_ids[int(idx)]
        results.append((pid, float(scores[int(idx)])))
    return results


def confidence_tier(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.45:
        return "MED"
    return "LOW"


app = FastAPI(title="CAIDE Matcher API")

print("Loading gallery...")
try:
    gallery_matrix, gallery_ids, metadata_map, photo_map = load_gallery(DATASET_ROOT)
    print("Server ready with", len(gallery_ids), "identities")
except Exception as e:
    print("ERROR loading dataset:", str(e))
    raise e
print(f"Loaded {len(gallery_ids)} identities.")
print("Server ready with", len(gallery_ids), "identities")

print("Loading InsightFace...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")


@app.get("/")
def root():
    return {"status": "ok", "message": "CAIDE matcher API is running"}


@app.post("/match")
async def match_face(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    faces = face_app.get(image)
    if not faces:
        return {
            "match_found": False,
            "message": "No face detected."
        }

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    query_emb = np.asarray(face.embedding, dtype=np.float32)
    query_emb = l2_normalize(query_emb)

    top3 = top_k_matches(query_emb, gallery_matrix, gallery_ids, k=3)

    best_id, best_score = top3[0]
    best_meta = metadata_map[best_id]

    best_candidate = {
        "identity_id": best_id,
        "name": best_meta.get("name", best_id),
        "score": round(best_score, 4),
        "confidence": confidence_tier(best_score),

        "gender": best_meta.get("gender", "N/A"),
        "domain": best_meta.get("domain", "N/A"),
        "nationality": best_meta.get("nationality", "N/A"),
        "date_of_birth": best_meta.get("date_of_birth", "N/A"),
        "profession": best_meta.get("profession", []),
        "known_for": best_meta.get("known_for", []),
        "affiliations": best_meta.get("affiliations", []),
        "text_profile": best_meta.get("text_profile", "N/A"),

        "photo_path": photo_map.get(best_id)
    }

    alternatives = []
    for pid, score in top3[1:]:
        meta = metadata_map[pid]
        alternatives.append({
            "identity_id": pid,
            "name": meta.get("name", pid),
            "score": round(score, 4),
            "confidence": confidence_tier(score)
        })

    accepted = best_score >= SIMILARITY_THRESHOLD

    return {
        "match_found": accepted,
        "best_candidate": best_candidate,
        "alternatives": alternatives,
        "threshold": SIMILARITY_THRESHOLD,
        "message": "Match accepted" if accepted else "Below threshold / unknown"
    }