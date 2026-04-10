import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


DATASET_ROOT = Path("CAIDE - DATA")  # change this
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES_PER_PERSON = 3


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def main() -> None:
    # Use CPUExecutionProvider if you do not have CUDA set up
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for person_dir in sorted(DATASET_ROOT.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("P"):
            continue

        image_files = sorted(
            [
                p for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        )[:MAX_IMAGES_PER_PERSON]

        if len(image_files) == 0:
            print(f"Skipping {person_dir.name}: no images found")
            continue

        if len(image_files) == 0:
            print(f"Skipping {person_dir.name}: no images found")
            continue

        results = []
        skipped = []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append({"image": img_path.name, "reason": "failed_to_read"})
                continue

            faces = app.get(img)
            if not faces:
                skipped.append({"image": img_path.name, "reason": "no_face_detected"})
                continue

            # pick the largest detected face
            best_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            embedding = np.asarray(best_face.embedding, dtype=np.float32)
            embedding = l2_normalize(embedding)

            results.append(
                {
                    "image": img_path.name,
                    "embedding_dim": int(embedding.shape[0]),
                    "embedding": embedding.tolist()
                }
            )

        output = {
            "id": person_dir.name,
            "model": "ArcFace",
            "num_requested_images": len(image_files),
            "num_saved_embeddings": len(results),
            "vectors": results,
            "skipped": skipped
        }

        out_path = person_dir / "arcface_vectors.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()