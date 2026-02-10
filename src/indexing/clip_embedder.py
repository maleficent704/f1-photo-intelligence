import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path
import chromadb
import sqlite3
import hashlib
import logging
from datetime import datetime

class CLIPEmbedder:
    def __init__(
        self,
        db_path: str = './data/f1_photos.db',
        chroma_path: str = './data/chroma_db',
        model_name: str = 'clip-ViT-L-14'
    ):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path

        # ChromaDB for vector search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="f1_photos_clip",
            metadata={"hnsw:space": "cosine"}
        )

        self._init_db()

    def _init_db(self):
        """Initialize SQLite tracking database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                grand_prix TEXT,

                -- Processing status
                clip_embedded_at DATETIME,
                vlm_described_at DATETIME,
                faces_scanned_at DATETIME,

                -- VLM output
                description TEXT,
                tags JSON,

                -- Metadata
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                exif_date DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT NOT NULL,
                person_id TEXT,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                encoding BLOB,
                FOREIGN KEY (photo_id) REFERENCES photos(id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS known_people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                team TEXT,
                reference_count INTEGER DEFAULT 0,
                metadata JSON
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS photo_tags (
                photo_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL,
                PRIMARY KEY (photo_id, tag, source),
                FOREIGN KEY (photo_id) REFERENCES photos(id)
            )
        ''')

        conn.commit()
        conn.close()

    def scan_photos(self, photo_dir: str) -> int:
        """Discover photos and register them in the database."""
        photo_dir = Path(photo_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}

        conn = sqlite3.connect(self.db_path)
        count = 0

        for img_path in photo_dir.rglob('*'):
            if img_path.suffix.lower() not in extensions:
                continue

            photo_id = hashlib.md5(str(img_path).encode()).hexdigest()

            # Extract grand prix from directory structure
            relative = img_path.relative_to(photo_dir)
            grand_prix = relative.parts[0] if len(relative.parts) > 1 else 'unknown'

            try:
                conn.execute('''
                    INSERT OR IGNORE INTO photos (id, file_path, grand_prix, file_size)
                    VALUES (?, ?, ?, ?)
                ''', (photo_id, str(img_path), grand_prix, img_path.stat().st_size))
                count += 1
            except Exception as e:
                logging.warning(f"Failed to register {img_path}: {e}")

        conn.commit()
        conn.close()
        return count

    def embed_batch(self, batch_size: int = 100, force: bool = False) -> int:
        """Embed unprocessed photos. Can be interrupted and resumed."""
        conn = sqlite3.connect(self.db_path)

        # Find unprocessed photos
        if force:
            rows = conn.execute('SELECT id, file_path FROM photos').fetchall()
        else:
            rows = conn.execute(
                'SELECT id, file_path FROM photos WHERE clip_embedded_at IS NULL'
            ).fetchall()

        logging.info(f"Found {len(rows)} photos to embed")

        processed = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            ids = []
            images = []
            metadatas = []

            for photo_id, file_path in batch:
                try:
                    img = Image.open(file_path).convert('RGB')

                    # Resize for consistency
                    img.thumbnail((512, 512))

                    ids.append(photo_id)
                    images.append(img)

                    # Extract grand prix from path
                    gp = Path(file_path).parent.name
                    metadatas.append({
                        'file_path': file_path,
                        'grand_prix': gp
                    })
                except Exception as e:
                    logging.warning(f"Failed to load {file_path}: {e}")
                    continue

            if not images:
                continue

            # Batch encode
            embeddings = self.model.encode(images, show_progress_bar=True)

            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )

            # Update SQLite tracking
            now = datetime.now().isoformat()
            for photo_id in ids:
                conn.execute(
                    'UPDATE photos SET clip_embedded_at = ? WHERE id = ?',
                    (now, photo_id)
                )

            conn.commit()
            processed += len(ids)
            logging.info(f"Embedded {processed}/{len(rows)} photos")

        conn.close()
        return processed

    def search(
        self,
        query: str,
        n_results: int = 20,
        grand_prix: str = None
    ) -> list[dict]:
        """Search photos using natural language."""

        # Encode the text query
        query_embedding = self.model.encode(query).tolist()

        # Build filter
        where = None
        if grand_prix:
            where = {"grand_prix": grand_prix}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return [
            {
                'photo_id': results['ids'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ]
