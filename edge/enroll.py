"""CLI enrollment: add a person to the gallery from photo files.

Usage (on the Pi or any machine with the ``cv`` extra installed):

    python -m edge.enroll --name "Demo person" --photos ./private-photos/

Accepts photo files and/or directories (jpg/jpeg/png). The best-scoring face
in each photo becomes one gallery embedding; ambiguous group photos are rejected.
Calibration and representative evaluation data are required for real use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edge.config import EdgeSettings
from edge.embedder import FaceEmbedder, InsightFaceEmbedder
from edge.store import HomeStore

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_image_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES)
        elif path.suffix.lower() in _IMAGE_SUFFIXES:
            files.append(path)
    return files


def enroll_photos(
    store: HomeStore,
    embedder: FaceEmbedder,
    *,
    name: str,
    photos: list[Path],
    is_owner: bool = False,
) -> tuple[int, list[str]]:
    """Embed each photo's best face into the gallery. Returns (ok, failed)."""
    import cv2  # lazy: only the real CLI path needs it

    person_id = store.add_person(name, is_owner=is_owner)
    enrolled = 0
    failed: list[str] = []
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            failed.append(f"{photo.name} (okunamadı)")
            continue
        faces = embedder.embed_all(image)
        if not faces:
            failed.append(f"{photo.name} (yüz bulunamadı)")
            continue
        if len(faces) != 1:
            failed.append(f"{photo.name} (exactly one face is required)")
            continue
        store.add_embedding(person_id, faces[0][0])
        enrolled += 1
    return enrolled, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enroll a person from photos.")
    parser.add_argument("--name", required=True, help="person's display name")
    parser.add_argument("--owner", action="store_true", help="mark as the home owner")
    parser.add_argument(
        "--photos", nargs="+", required=True, type=Path, help="photo files and/or directories"
    )
    args = parser.parse_args(argv)

    settings = EdgeSettings()
    files = collect_image_files(list(args.photos))
    if not files:
        parser.error("no image files found under --photos")

    store = HomeStore(settings.data_dir)
    embedder = InsightFaceEmbedder(settings.insightface_model, model_root=settings.insightface_root)
    enrolled, failed = enroll_photos(
        store, embedder, name=args.name, photos=files, is_owner=args.owner
    )
    print(f"{args.name}: {enrolled} embedding(s) enrolled.")
    for item in failed:
        print(f"  skipped: {item}")
    store.close()
    return 0 if enrolled else 1


if __name__ == "__main__":
    raise SystemExit(main())
