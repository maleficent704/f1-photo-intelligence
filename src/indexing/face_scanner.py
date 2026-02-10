import face_recognition
import numpy as np
import sqlite3
import json
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime

class FaceScanner:
    def __init__(
        self,
        db_path: str = './data/f1_photos.db',
        reference_dir: str = './data/face_references',
        tolerance: float = 0.55
    ):
        self.db_path = db_path
        self.reference_dir = Path(reference_dir)
        self.tolerance = tolerance
        self.known_encodings = {}
        self.known_names = {}

        self._load_references()

    def _load_references(self):
        """Load reference face encodings for known people."""
        if not self.reference_dir.exists():
            self.reference_dir.mkdir(parents=True)
            logging.info(f"Created reference directory at {self.reference_dir}")
            logging.info("Add subdirectories with reference photos: {name}/photo1.jpg")
            return

        for person_dir in self.reference_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            encodings = []

            for img_path in person_dir.glob('*'):
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue

                try:
                    img = face_recognition.load_image_file(str(img_path))
                    face_encs = face_recognition.face_encodings(img)

                    if face_encs:
                        encodings.append(face_encs[0])
                        logging.info(f"Loaded reference for {person_id}: {img_path.name}")
                    else:
                        logging.warning(f"No face found in reference {img_path}")

                except Exception as e:
                    logging.warning(f"Failed to load reference {img_path}: {e}")

            if encodings:
                self.known_encodings[person_id] = encodings
                # Average encoding for faster comparison
                self.known_names[person_id] = np.mean(encodings, axis=0)
                logging.info(f"Loaded {len(encodings)} references for {person_id}")

        logging.info(f"Loaded references for {len(self.known_encodings)} people")

    def add_reference_person(self, person_id: str, name: str, category: str = 'driver', team: str = None):
        """Register a person in the database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO known_people (id, name, category, team)
            VALUES (?, ?, ?, ?)
        ''', (person_id, name, category, team))
        conn.commit()
        conn.close()

    def scan_photo(self, file_path: str) -> list[dict]:
        """Detect and identify faces in a photo."""

        img = face_recognition.load_image_file(file_path)

        # Detect faces
        face_locations = face_recognition.face_locations(img, model='hog')

        if not face_locations:
            return []

        # Encode detected faces
        face_encodings = face_recognition.face_encodings(img, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            top, right, bottom, left = location

            # Try to match against known people
            best_match = None
            best_distance = float('inf')

            for person_id, ref_encodings in self.known_encodings.items():
                distances = face_recognition.face_distance(ref_encodings, encoding)
                min_distance = min(distances)

                if min_distance < self.tolerance and min_distance < best_distance:
                    best_match = person_id
                    best_distance = min_distance

            results.append({
                'person_id': best_match,
                'confidence': 1.0 - best_distance if best_match else 0.0,
                'bbox': {'x': left, 'y': top, 'w': right - left, 'h': bottom - top},
                'encoding': encoding
            })

        return results

    def process_batch(self, limit: int = None, use_gpu: bool = False) -> int:
        """Scan unprocessed photos for faces. Resumable."""
        conn = sqlite3.connect(self.db_path)

        query = 'SELECT id, file_path FROM photos WHERE faces_scanned_at IS NULL'
        if limit:
            query += f' LIMIT {limit}'

        rows = conn.execute(query).fetchall()
        logging.info(f"Found {len(rows)} photos to scan for faces")

        processed = 0
        for photo_id, file_path in rows:
            try:
                faces = self.scan_photo(file_path)

                for face in faces:
                    conn.execute('''
                        INSERT INTO face_detections
                        (photo_id, person_id, confidence, bbox_x, bbox_y, bbox_w, bbox_h, encoding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        photo_id,
                        face['person_id'],
                        face['confidence'],
                        face['bbox']['x'],
                        face['bbox']['y'],
                        face['bbox']['w'],
                        face['bbox']['h'],
                        face['encoding'].tobytes()
                    ))

                    # Add tag for identified person
                    if face['person_id']:
                        conn.execute('''
                            INSERT OR IGNORE INTO photo_tags (photo_id, tag, source, confidence)
                            VALUES (?, ?, 'face', ?)
                        ''', (photo_id, f"person:{face['person_id']}", face['confidence']))

                conn.execute(
                    'UPDATE photos SET faces_scanned_at = ? WHERE id = ?',
                    (datetime.now().isoformat(), photo_id)
                )

                conn.commit()
                processed += 1

                if processed % 100 == 0:
                    logging.info(f"Scanned {processed}/{len(rows)} photos")

            except Exception as e:
                logging.error(f"Failed to scan {file_path}: {e}")
                continue

        conn.close()
        return processed
