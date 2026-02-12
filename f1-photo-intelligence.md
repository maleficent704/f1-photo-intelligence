# F1 Photo Intelligence Pipeline

## Overview

An AI-powered system for automatically tagging, describing, and searching a personal F1 photo collection (~12k images organized by Grand Prix). The goal: go from folders of unsorted race photos to a richly searchable archive where you can find things like "Lestappen moments," "wet race starts," "Ferrari pit stops," or "podium celebrations at Monza" — all without manually tagging a single photo.

## What This System Does

Three AI layers work together to make your photos searchable:

1. **CLIP embeddings** — Every photo gets converted to a vector that understands both visual content and natural language. Search with plain English, get matching photos. Fast, zero-shot, no training needed.

2. **Vision LLM descriptions** — A local vision model (Qwen2-VL or LLaVA) looks at each photo and writes a rich text description: teams, drivers, track features, weather, race phase, crowd shots, candid moments. These descriptions become searchable text and structured tags.

3. **Face recognition** — Reference photos of specific people (drivers, team principals, your own family at races) enable finding every photo containing that person, regardless of context. This is how you find every Lestappen photo without manually tagging any of them.

## Why These Three Layers

Each layer catches things the others miss:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Query Examples                                   │
├──────────────────────────┬──────────────────────────────────────────────┤
│  "red car on track"      │  CLIP handles this perfectly                │
│  "Ferrari SF-24 on       │  Vision LLM identifies specific car model   │
│   mediums at Silverstone" │  and can read tire compound colors          │
│  "Max and Charles         │  Face recognition finds both people,       │
│   standing together"      │  CLIP/LLM confirms proximity               │
│  "rainy qualifying"       │  CLIP + LLM both contribute                │
│  "photos with me in them" │  Face recognition with your reference pics │
└──────────────────────────┴──────────────────────────────────────────────┘
```

CLIP is fast and fuzzy — great for vibes-based search. Vision LLM is slow but detailed — great for structured metadata. Face recognition solves the "who" problem that neither of the others handle well when helmets are off.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Photo Source                                   │
│         ~/Photos/F1/{grand_prix_name}/*.jpg                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Batch Processing Pipeline                       │
│                  (runs on RTX 4080)                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    CLIP      │  │  Vision LLM  │  │    Face      │          │
│  │  Embedding   │  │  Description │  │  Recognition │          │
│  │  (~1 sec/img)│  │ (~5-10s/img) │  │  (~0.5s/img) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         └─────────────────┼──────────────────┘                   │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │  Metadata Store │                             │
│                  │    (SQLite)     │                             │
│                  └────────┬────────┘                             │
│                           │                                      │
│                  ┌─────────────────┐                             │
│                  │  Vector Store   │                             │
│                  │   (ChromaDB)    │                             │
│                  └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Query Interface                              │
│                                                                  │
│  Natural language search  →  Matching photos + metadata         │
│  Tag-based filtering      →  Structured queries                 │
│  Face search              →  "Photos with this person"          │
│  Combined                 →  "Max at Monaco in the rain"        │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

All processing runs on the main PC with the RTX 4080.

**CLIP embedding pass:** ~12k images × ~1 second each = ~3-4 hours. GPU memory usage is light (~2GB), can run alongside other things.

**Vision LLM pass:** ~12k images × ~5-10 seconds each = ~17-33 hours. This is the heavy one. Run overnight or across a couple of sessions. Uses most of the 16GB VRAM. Can be interrupted and resumed — tracks which images are already processed.

**Face recognition:** ~12k images × ~0.5 seconds each = ~1.5-2 hours. Primarily CPU-bound, GPU optional.

**Storage:** Metadata DB and embeddings will be small — maybe 500MB total for 12k images. The photos themselves stay where they are.

---

## Layer 1: CLIP Embeddings

### What CLIP Does

CLIP (Contrastive Language-Image Pretraining) learned to map images and text into the same vector space. An image of a red car and the text "red car" end up near each other in vector space. This means you can search images using plain English without any training or labeling.

### What CLIP Is Good At (Zero-Shot)

CLIP already understands, with no F1-specific training:
- Car colors and team liveries ("red car," "blue and yellow car")
- Weather conditions ("rain," "overcast," "sunny")
- Scene types ("podium," "pit lane," "starting grid," "crowd")
- General actions ("celebrating," "driving," "standing," "spraying champagne")
- Objects ("trophy," "helmet," "tire," "steering wheel")
- Settings ("night race," "street circuit," "grandstand")

### What CLIP Struggles With

- Specific driver identification (can't tell Max from Charles by face)
- Reading text on cars/signs (sponsor logos, car numbers)
- Distinguishing F1-specific concepts ("medium tires" vs "hard tires" by color)
- Exact track identification without obvious landmarks
- Temporal context ("qualifying" vs "race" from a still image)

### Implementation

```python
# src/indexing/clip_embedder.py

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
        model_name: str = 'clip-ViT-L-14'  # Larger = better quality
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
                person_id TEXT,           -- FK to known_people, NULL if unknown
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                encoding BLOB,            -- Face encoding for matching
                FOREIGN KEY (photo_id) REFERENCES photos(id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS known_people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,            -- 'driver', 'team_principal', 'personal', etc.
                team TEXT,
                reference_count INTEGER DEFAULT 0,
                metadata JSON
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photo_tags (
                photo_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                source TEXT NOT NULL,      -- 'clip', 'vlm', 'face', 'manual'
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
            # Assumes: ~/Photos/F1/{grand_prix_name}/image.jpg
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
                    
                    # Resize for consistency (CLIP works fine at 224x224
                    # but larger preserves detail for later use)
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
```

### CLIP Model Selection

| Model | VRAM | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| `clip-ViT-B-32` | ~1GB | Fastest | Good | Fine for initial testing |
| `clip-ViT-L-14` | ~2GB | Medium | Better | Recommended for production |
| `clip-ViT-H-14` | ~3GB | Slower | Best | Worth it if you want maximum quality |
| `openclip-ViT-bigG-14` | ~5GB | Slowest | Highest | OpenCLIP variant, significant quality jump |

Start with `clip-ViT-L-14` — good balance of quality and speed. You can always re-embed with a better model later since the pipeline tracks what's been processed.

---

## Layer 2: Vision LLM Descriptions

### What This Adds Over CLIP

CLIP gives you fuzzy semantic search but no structured data. The vision LLM pass looks at each photo and generates:

- **Free-text description** — "Two Ferrari drivers on the podium at Monza, Leclerc holding trophy, rain-dampened track visible in background"
- **Structured tags** — team, track, conditions, race phase, car number, tire compound
- **F1-specific knowledge** — Can identify liveries by season, read car numbers, recognize track layouts, understand race context

### Model Options (Ollama)

| Model | Size | VRAM | Quality | Speed |
|-------|------|------|---------|-------|
| `llava:7b` | 7B | ~5GB | Good | ~3s/image |
| `llava:13b` | 13B | ~9GB | Better | ~6s/image |
| `qwen2-vl:7b` | 7B | ~5GB | Very good | ~4s/image |
| `minicpm-v:8b` | 8B | ~6GB | Good | ~4s/image |

**Recommendation:** `qwen2-vl:7b` — it's excellent at reading text in images (car numbers, sponsor logos) and has strong scene understanding. If you want even better results and can spare the VRAM, `llava:13b` or waiting for larger Qwen2-VL variants.

### Implementation

```python
# src/indexing/vlm_describer.py

import requests
import base64
import json
import sqlite3
from pathlib import Path
from PIL import Image
import io
import logging
from datetime import datetime

class VLMDescriber:
    def __init__(
        self,
        db_path: str = './data/f1_photos.db',
        ollama_url: str = 'http://localhost:11434',
        model: str = 'qwen2-vl:7b'
    ):
        self.db_path = db_path
        self.ollama_url = ollama_url
        self.model = model
    
    def _image_to_base64(self, file_path: str, max_size: int = 1024) -> str:
        """Load and resize image, return base64."""
        img = Image.open(file_path).convert('RGB')
        img.thumbnail((max_size, max_size))
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def describe_photo(self, file_path: str) -> dict:
        """Get detailed F1-specific description from vision model."""
        
        img_b64 = self._image_to_base64(file_path)
        
        prompt = """Analyze this Formula 1 photograph. Provide:

1. DESCRIPTION: A detailed description of what's happening in the image (2-3 sentences).

2. TAGS: Provide structured tags in these categories (leave blank if not applicable):
- teams: Which F1 teams are visible (e.g., Ferrari, Red Bull, McLaren)
- drivers: Any identifiable drivers (by car number, helmet, or recognition)
- track: Which circuit this might be (if identifiable)
- session: What session this appears to be (practice, qualifying, race, podium, pit lane, paddock, press conference, fan zone)
- conditions: Weather and track conditions (dry, wet, overcast, night, sunset)
- action: What's happening (on-track battle, pit stop, celebration, crash, start, formation lap)
- car_numbers: Any visible car numbers
- tire_compound: Visible tire compounds (soft/red, medium/yellow, hard/white, intermediate/green, wet/blue)
- objects: Notable objects (trophy, champagne, flag, safety car, medical car)
- people_count: Approximate number of people prominently visible
- composition: Photo type (close-up, wide shot, aerial, cockpit, behind-the-scenes)

Respond in JSON format only, no other text:
{
    "description": "...",
    "tags": {
        "teams": [],
        "drivers": [],
        "track": "",
        "session": "",
        "conditions": [],
        "action": [],
        "car_numbers": [],
        "tire_compound": [],
        "objects": [],
        "people_count": 0,
        "composition": ""
    }
}"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent tagging
                    "num_predict": 1024
                }
            }
        )
        
        result = response.json()['response']
        
        # Parse JSON from response (handle potential markdown wrapping)
        try:
            # Strip markdown code fences if present
            cleaned = result.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1]
                cleaned = cleaned.rsplit('```', 1)[0]
            
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse VLM response for {file_path}")
            return {
                "description": result,  # Store raw response
                "tags": {},
                "parse_error": True
            }
    
    def process_batch(self, limit: int = None) -> int:
        """Process unprocessed photos through VLM. Resumable."""
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT id, file_path FROM photos WHERE vlm_described_at IS NULL'
        if limit:
            query += f' LIMIT {limit}'
        
        rows = conn.execute(query).fetchall()
        logging.info(f"Found {len(rows)} photos to describe")
        
        processed = 0
        for photo_id, file_path in rows:
            try:
                result = self.describe_photo(file_path)
                
                # Store description and tags
                conn.execute('''
                    UPDATE photos 
                    SET description = ?, tags = ?, vlm_described_at = ?
                    WHERE id = ?
                ''', (
                    result.get('description', ''),
                    json.dumps(result.get('tags', {})),
                    datetime.now().isoformat(),
                    photo_id
                ))
                
                # Also store individual tags for easy querying
                tags = result.get('tags', {})
                for category, values in tags.items():
                    if isinstance(values, list):
                        for v in values:
                            conn.execute('''
                                INSERT OR IGNORE INTO photo_tags (photo_id, tag, source, confidence)
                                VALUES (?, ?, 'vlm', 1.0)
                            ''', (photo_id, f"{category}:{v}"))
                    elif values:  # Non-empty string/number
                        conn.execute('''
                            INSERT OR IGNORE INTO photo_tags (photo_id, tag, source, confidence)
                            VALUES (?, ?, 'vlm', 1.0)
                        ''', (photo_id, f"{category}:{values}"))
                
                conn.commit()
                processed += 1
                
                if processed % 100 == 0:
                    logging.info(f"Described {processed}/{len(rows)} photos")
                    
            except Exception as e:
                logging.error(f"Failed to describe {file_path}: {e}")
                continue
        
        conn.close()
        return processed
```

### VLM Prompt Tuning Tips

The prompt above is a starting point. After running it on a sample batch, you'll likely want to adjust:

- **Team identification accuracy** — If it's confusing Aston Martin green with other greens, add hints about current livery colors
- **Track identification** — Some tracks are very recognizable (Monaco, Singapore night race, Spa), others less so. You might want to add the Grand Prix folder name as a hint
- **Session detection** — The model might struggle distinguishing qualifying from race in some shots. Consider if you even need this distinction
- **JSON reliability** — If the model frequently breaks JSON format, switch to a simpler output format or add retry logic

### Enriching with Grand Prix Context

Since your photos are already organized by GP, you can inject that context:

```python
def describe_photo_with_context(self, file_path: str, grand_prix: str) -> dict:
    """Add GP context to help the model."""
    
    # Map folder names to useful context
    gp_context = {
        'monaco-2024': 'This photo is from the 2024 Monaco Grand Prix at Circuit de Monaco, a street circuit.',
        'silverstone-2024': 'This photo is from the 2024 British Grand Prix at Silverstone Circuit.',
        'spa-2024': 'This photo is from the 2024 Belgian Grand Prix at Spa-Francorchamps.',
        # ... add your folder names
    }
    
    context = gp_context.get(grand_prix, f'This photo is from the {grand_prix} Grand Prix.')
    
    # Prepend context to the prompt
    prompt = f"Context: {context}\n\n{self.base_prompt}"
    # ...
```

---

## Layer 3: Face Recognition

### Why This Layer Matters

CLIP can find "two people celebrating" and the VLM might identify team uniforms, but neither reliably identifies specific individuals — especially in casual/candid shots where they're not in team gear. Face recognition solves the "who" problem.

### The Lestappen Use Case

To find all Max + Charles content:

1. Provide 5-10 reference photos of each person (clear face shots)
2. System encodes these as reference face vectors
3. For each photo in the collection, detect all faces and compare to references
4. Tag photos with detected people
5. Query: photos containing both "max_verstappen" AND "charles_leclerc"

This catches everything — podium moments, press conferences, driver parade, paddock walks, even fan meetups if you happened to catch them.

### Implementation

```python
# src/indexing/face_scanner.py

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
        tolerance: float = 0.55  # Lower = stricter matching
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
            
            person_id = person_dir.name  # e.g., 'max_verstappen'
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
        face_locations = face_recognition.face_locations(img, model='hog')  # 'cnn' for GPU
        
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
```

### Reference Photo Directory Structure

```
data/face_references/
├── max_verstappen/
│   ├── headshot_1.jpg        # Clear front-facing
│   ├── profile.jpg           # Side angle
│   ├── sunglasses.jpg        # With accessories
│   ├── candid_1.jpg          # Natural setting
│   └── podium.jpg            # In race suit
│
├── charles_leclerc/
│   ├── headshot_1.jpg
│   ├── headshot_2.jpg
│   └── ...
│
├── lando_norris/
│   └── ...
│
├── kelly/                    # Your own face for personal photos
│   └── ...
│
└── sam/                      # Family members at races
    └── ...
```

**Tips for good reference photos:**
- 5-10 per person is ideal, more doesn't help much
- Include different angles, lighting, accessories (sunglasses, hats)
- Include photos with and without team gear
- Clear face visibility is more important than high resolution
- The `face_recognition` library works best with front-facing or 3/4 angle shots

### Face Recognition Model Options

The `face_recognition` library uses dlib under the hood, which is solid but there are alternatives:

| Library | Speed | Accuracy | GPU | Notes |
|---------|-------|----------|-----|-------|
| `face_recognition` (dlib) | Medium | Good | Optional | Simplest to set up, well-documented |
| `deepface` | Slower | Better | Yes | Multiple backends (VGGFace, FaceNet, ArcFace) |
| `insightface` | Fastest | Best | Yes | Industry standard, ArcFace model |

`face_recognition` is the pragmatic starting point. If matching accuracy becomes an issue (misidentifying drivers who look similar), consider `insightface` with ArcFace — it's significantly more accurate but requires more setup.

---

## Query Interface

### Combined Search

This is where all three layers come together:

```python
# src/query/photo_search.py

import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

class F1PhotoSearch:
    def __init__(
        self,
        db_path: str = './data/f1_photos.db',
        chroma_path: str = './data/chroma_db',
        clip_model: str = 'clip-ViT-L-14'
    ):
        self.db_path = db_path
        self.clip_model = SentenceTransformer(clip_model)
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.clip_collection = self.chroma_client.get_collection("f1_photos_clip")
    
    def search(
        self,
        query: str = None,
        people: list[str] = None,
        tags: list[str] = None,
        grand_prix: str = None,
        require_all_people: bool = True,
        n_results: int = 20
    ) -> list[dict]:
        """
        Combined search across all layers.
        
        Examples:
            search(people=['max_verstappen', 'charles_leclerc'])  # Lestappen!
            search(query='rainy race start')
            search(query='podium celebration', grand_prix='monza-2024')
            search(query='pit stop', tags=['teams:Ferrari'])
            search(people=['charles_leclerc'], query='celebrating')
        """
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Start with all photos
        candidate_ids = None
        
        # Filter by people (face recognition layer)
        if people:
            for person_id in people:
                person_photos = set(
                    row[0] for row in conn.execute('''
                        SELECT DISTINCT photo_id FROM face_detections
                        WHERE person_id = ? AND confidence > 0.4
                    ''', (person_id,)).fetchall()
                )
                
                if candidate_ids is None:
                    candidate_ids = person_photos
                elif require_all_people:
                    candidate_ids &= person_photos  # Intersection
                else:
                    candidate_ids |= person_photos  # Union
        
        # Filter by structured tags (VLM layer)
        if tags:
            for tag in tags:
                tag_photos = set(
                    row[0] for row in conn.execute('''
                        SELECT photo_id FROM photo_tags
                        WHERE tag = ?
                    ''', (tag,)).fetchall()
                )
                
                if candidate_ids is None:
                    candidate_ids = tag_photos
                else:
                    candidate_ids &= tag_photos
        
        # Filter by grand prix
        if grand_prix:
            gp_photos = set(
                row[0] for row in conn.execute('''
                    SELECT id FROM photos WHERE grand_prix = ?
                ''', (grand_prix,)).fetchall()
            )
            
            if candidate_ids is None:
                candidate_ids = gp_photos
            else:
                candidate_ids &= gp_photos
        
        # If we have a text query, use CLIP for semantic ranking
        if query:
            query_embedding = self.clip_model.encode(query).tolist()
            
            # If we have candidates from other filters, search within them
            if candidate_ids:
                where_filter = {"photo_id": {"$in": list(candidate_ids)}} if len(candidate_ids) <= 1000 else None
                clip_results = self.clip_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * 3, len(candidate_ids)),  # Over-fetch for filtering
                    where=where_filter
                )
                
                # Rank by CLIP similarity, filtered to candidates
                ranked = []
                for i, photo_id in enumerate(clip_results['ids'][0]):
                    if candidate_ids is None or photo_id in candidate_ids:
                        ranked.append({
                            'photo_id': photo_id,
                            'clip_score': 1.0 - clip_results['distances'][0][i],
                            'metadata': clip_results['metadatas'][0][i]
                        })
                
                results = sorted(ranked, key=lambda x: x['clip_score'], reverse=True)[:n_results]
            else:
                # Pure CLIP search
                clip_results = self.clip_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                results = [
                    {
                        'photo_id': clip_results['ids'][0][i],
                        'clip_score': 1.0 - clip_results['distances'][0][i],
                        'metadata': clip_results['metadatas'][0][i]
                    }
                    for i in range(len(clip_results['ids'][0]))
                ]
        else:
            # No text query — just return filtered results
            if candidate_ids is None:
                candidate_ids = set(
                    row[0] for row in conn.execute('SELECT id FROM photos').fetchall()
                )
            
            results = [
                {'photo_id': pid, 'clip_score': None}
                for pid in list(candidate_ids)[:n_results]
            ]
        
        # Enrich results with full metadata
        for result in results:
            photo = conn.execute(
                'SELECT * FROM photos WHERE id = ?', (result['photo_id'],)
            ).fetchone()
            
            if photo:
                result['file_path'] = photo['file_path']
                result['grand_prix'] = photo['grand_prix']
                result['description'] = photo['description']
                result['tags'] = json.loads(photo['tags']) if photo['tags'] else {}
            
            # Get detected people
            faces = conn.execute('''
                SELECT person_id, confidence FROM face_detections
                WHERE photo_id = ? AND person_id IS NOT NULL
            ''', (result['photo_id'],)).fetchall()
            
            result['people'] = [
                {'person_id': f['person_id'], 'confidence': f['confidence']}
                for f in faces
            ]
        
        conn.close()
        return results
    
    # Convenience methods
    
    def lestappen(self, query: str = None, n_results: int = 20) -> list[dict]:
        """Find Lestappen content. Obviously the most important query."""
        return self.search(
            query=query,
            people=['max_verstappen', 'charles_leclerc'],
            require_all_people=True,
            n_results=n_results
        )
    
    def team_photos(self, team: str, query: str = None, n_results: int = 20) -> list[dict]:
        """Find photos of a specific team."""
        return self.search(
            query=query,
            tags=[f'teams:{team}'],
            n_results=n_results
        )
    
    def at_race(self, grand_prix: str, query: str = None, n_results: int = 20) -> list[dict]:
        """Find photos from a specific race."""
        return self.search(
            query=query,
            grand_prix=grand_prix,
            n_results=n_results
        )
```

### Example Queries

```python
search = F1PhotoSearch()

# The important one
search.lestappen()                                    # All Max+Charles photos
search.lestappen(query='podium')                      # Lestappen podium moments
search.lestappen(query='laughing')                     # Lestappen being adorable

# Team searches
search.team_photos('Ferrari')                          # All Ferrari content
search.team_photos('Ferrari', query='pit stop')        # Ferrari pit stops

# Scene searches
search.search(query='rainy race start')                # Dramatic wet starts
search.search(query='champagne celebration')           # Podium celebrations
search.search(query='crash barrier gravel')            # Dramatic moments
search.search(query='sunset over circuit')             # Scenic shots

# Combined
search.search(                                         # Charles celebrating at Monza
    query='celebrating',
    people=['charles_leclerc'],
    grand_prix='monza-2024'
)

search.search(                                         # Your family at Silverstone
    people=['kelly', 'sam'],
    grand_prix='silverstone-2024'
)

# Tag-based
search.search(tags=['conditions:wet', 'session:race']) # Wet races
search.search(tags=['tire_compound:soft'])              # Close-ups showing softs
```

---

## Processing Pipeline

### Full Pipeline Runner

```python
# src/pipeline/process_photos.py

import argparse
import logging
from datetime import datetime
from indexing.clip_embedder import CLIPEmbedder
from indexing.vlm_describer import VLMDescriber
from indexing.face_scanner import FaceScanner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/photo_pipeline.log'),
        logging.StreamHandler()
    ]
)

def run_pipeline(
    photo_dir: str,
    skip_clip: bool = False,
    skip_vlm: bool = False,
    skip_faces: bool = False,
    vlm_limit: int = None
):
    """Run the full processing pipeline."""
    
    db_path = './data/f1_photos.db'
    
    # Step 1: Discover photos
    logging.info("Step 1: Scanning for new photos...")
    embedder = CLIPEmbedder(db_path=db_path)
    new_photos = embedder.scan_photos(photo_dir)
    logging.info(f"Found {new_photos} photos")
    
    # Step 2: CLIP embeddings (fast)
    if not skip_clip:
        logging.info("Step 2: Generating CLIP embeddings...")
        embedded = embedder.embed_batch()
        logging.info(f"Embedded {embedded} photos")
    
    # Step 3: Face recognition (medium speed)
    if not skip_faces:
        logging.info("Step 3: Scanning for faces...")
        scanner = FaceScanner(db_path=db_path)
        scanned = scanner.process_batch()
        logging.info(f"Scanned {scanned} photos for faces")
    
    # Step 4: VLM descriptions (slow - do last)
    if not skip_vlm:
        logging.info("Step 4: Generating VLM descriptions...")
        describer = VLMDescriber(db_path=db_path)
        described = describer.process_batch(limit=vlm_limit)
        logging.info(f"Described {described} photos")
    
    logging.info("Pipeline complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1 Photo Intelligence Pipeline')
    parser.add_argument('photo_dir', help='Path to F1 photos directory')
    parser.add_argument('--skip-clip', action='store_true')
    parser.add_argument('--skip-vlm', action='store_true')
    parser.add_argument('--skip-faces', action='store_true')
    parser.add_argument('--vlm-limit', type=int, help='Limit VLM processing count')
    
    args = parser.parse_args()
    run_pipeline(
        args.photo_dir,
        skip_clip=args.skip_clip,
        skip_vlm=args.skip_vlm,
        skip_faces=args.skip_faces,
        vlm_limit=args.vlm_limit
    )
```

### Recommended Processing Order

**First run (getting something working fast):**

1. `python process_photos.py ~/Photos/F1 --skip-vlm --skip-faces` — Just CLIP. Takes ~3-4 hours but gives you searchable photos immediately.

2. Test CLIP search to make sure it's working and useful.

3. `python process_photos.py ~/Photos/F1 --skip-clip --skip-vlm` — Add face recognition. Takes ~2 hours. Now you can search by person.

4. `python process_photos.py ~/Photos/F1 --skip-clip --skip-faces --vlm-limit 500` — Start VLM descriptions on a subset. Check quality, adjust prompt.

5. `python process_photos.py ~/Photos/F1 --skip-clip --skip-faces` — Full VLM pass overnight.

**Adding new photos later:**

The pipeline is incremental — it only processes photos that haven't been processed yet. Just dump new race photos into the appropriate GP folder and re-run.

---

## Directory Structure

```
~/projects/f1-photo-intelligence/     # Git repo
├── src/
│   ├── indexing/
│   │   ├── clip_embedder.py
│   │   ├── vlm_describer.py
│   │   └── face_scanner.py
│   ├── query/
│   │   └── photo_search.py
│   ├── pipeline/
│   │   └── process_photos.py
│   └── ui/                           # Future: web interface
│       └── ...
│
├── config/
│   └── pipeline_config.yaml
│
├── requirements.txt
├── README.md
└── .gitignore

~/Photos/F1/                          # Photo storage (NOT in repo)
├── bahrain-2024/
├── jeddah-2024/
├── australia-2024/
├── monaco-2024/
├── silverstone-2024/
├── monza-2024/
└── ...

~/projects/f1-photo-intelligence/data/  # Generated data (NOT in repo)
├── f1_photos.db                      # SQLite metadata
├── chroma_db/                        # CLIP embeddings
├── face_references/                  # Reference photos for face recognition
│   ├── max_verstappen/
│   ├── charles_leclerc/
│   ├── lando_norris/
│   ├── kelly/
│   └── sam/
└── logs/
```

### Requirements

```
# requirements.txt

# CLIP embeddings
sentence-transformers>=2.2.0
torch>=2.0.0

# Vector store
chromadb>=0.4.0

# Face recognition
face_recognition>=1.3.0
dlib>=19.24.0          # May need cmake installed first

# Image processing
Pillow>=10.0.0

# HTTP (for Ollama)
requests>=2.31.0

# Utilities
numpy>=1.24.0
```

### Installation Notes

```bash
# face_recognition requires dlib which needs cmake
# On Mac:
brew install cmake
pip install face_recognition

# On Linux:
sudo apt install cmake
pip install face_recognition

# If dlib install fails, there's a prebuilt wheel:
pip install dlib-bin  # Unofficial but works

# For GPU-accelerated face detection (optional):
# dlib needs to be built with CUDA support - see dlib docs
```

---

## Future Enhancements

### Web UI for Browsing

A simple web interface to browse and search photos would be much nicer than command-line queries. Could be a Flask/FastAPI app that serves a grid of photo thumbnails with a search bar, or a React frontend if you want to get fancy. This would also make it easy to manually correct tags and face identifications.

### Auto-Tagging New Photos

Set up a file watcher that detects new photos in the F1 directory and automatically queues them for processing. Combined with the MINIX running the pipeline on schedule, new race photos would be searchable within hours of being added.

### Integration with Personal Data Pipeline

Cross-reference with calendar data — if you have calendar events for races you attended, the system could automatically tag those GPs as "attended in person" vs "from broadcast/social media."

### Smart Albums / Auto-Curation

Use the search system to automatically generate "albums" like:
- Best Lestappen moments of 2024
- Most dramatic race starts
- Best scenic/artistic shots
- All podium celebrations by driver

### Duplicate Detection

CLIP embeddings can also find near-duplicate photos (burst shots, similar angles). Could flag these for cleanup.

### Model Improvements Over Time

Vision models are improving rapidly. The pipeline is designed so you can re-run the VLM pass with a better model later and overwrite the old descriptions. The `vlm_described_at` timestamp lets you selectively reprocess old descriptions.

---

## Getting Started Checklist

1. **Set up the project**
   - [ ] Create `~/projects/f1-photo-intelligence/` with directory structure
   - [ ] Set up Python environment and install requirements
   - [ ] Verify photo directory structure

2. **CLIP pass (get searching fast)**
   - [ ] Run scan to register all photos
   - [ ] Run CLIP embedding batch
   - [ ] Test a few searches to verify it works

3. **Face recognition setup**
   - [ ] Collect 5-10 reference photos per person of interest
   - [ ] Organize into `face_references/{person_id}/` directories
   - [ ] Register people in database
   - [ ] Run face scanning batch
   - [ ] Test: search for specific people

4. **VLM descriptions**
   - [ ] Pull vision model in Ollama (`ollama pull qwen2-vl:7b`)
   - [ ] Test on ~10 photos, check description quality
   - [ ] Adjust prompt if needed
   - [ ] Run full batch (overnight)

5. **Combined queries**
   - [ ] Test combined searches (people + text + tags)
   - [ ] Test the `lestappen()` method, obviously
   - [ ] Verify results make sense

6. **Optional: incremental processing**
   - [ ] Set up for new photos to be auto-detected
   - [ ] Test adding a new GP folder and reprocessing
