# F1 Photo Intelligence - Setup & Usage Guide

Complete reference for setting up and using the F1 Photo Intelligence system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Initial Setup](#initial-setup)
3. [Processing Pipeline](#processing-pipeline)
4. [Searching Photos](#searching-photos)
5. [Understanding the Layers](#understanding-the-layers)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

The F1 Photo Intelligence system has **three AI layers** that work together:

### Layer 1: CLIP Embeddings ✅ (Completed)
**What it does:** Converts photos to vectors that understand natural language
**Speed:** ~0.4 seconds per photo (CPU), ~1 second (GPU)
**Purpose:** Enable natural language search
**Example:** Search "red car on track" → finds all Ferrari photos

### Layer 2: Vision LLM (In Progress)
**What it does:** Analyzes each photo and generates F1-specific descriptions and tags
**Speed:** ~5-10 seconds per photo
**Purpose:** Structured metadata, detailed descriptions
**Example:** Identifies "Ferrari SF-24 on medium tires at Silverstone, qualifying session"

### Layer 3: Face Recognition (Optional)
**What it does:** Identifies specific people in photos
**Speed:** ~0.5 seconds per photo
**Purpose:** Find photos with specific drivers/people
**Requirement:** CMake installed (Windows issue - can add later)

---

## Initial Setup

### 1. Project Structure
```
f1-photo-intelligence/
├── src/
│   ├── indexing/          # Processing modules
│   ├── query/             # Search interface
│   └── pipeline/          # Pipeline runner
├── data/                  # Generated data (not in git)
│   ├── f1_photos.db      # SQLite metadata
│   ├── chroma_db/        # CLIP embeddings
│   └── face_references/  # Reference photos (optional)
├── venv/                 # Python virtual environment
└── requirements.txt      # Dependencies
```

### 2. Dependencies

**Installed:**
- Python 3.12
- PyTorch 2.10.0 (CPU version)
- sentence-transformers (CLIP)
- ChromaDB (vector database)
- Ollama (for vision LLM)

**Not installed:**
- face_recognition (requires CMake - optional)
- dlib (requires CMake - optional)

### 3. Photo Organization

**Your photos:** `C:\Users\kelly\OneDrive\Pictures\f1 photos\Formula 1-3-001\Formula 1\`

**Filename format:** `YYYY GP_Name GP - Description.jpg`
**Example:** `2021 Abu Dhabi GP - Carlos Sainz.jpg`

**Metadata extraction:**
- Year: `2021`
- Grand Prix: `abu-dhabi-2021` (normalized)
- Keeps photos in original location (no copying/moving)

---

## Processing Pipeline

### Layer 1: CLIP Embeddings (COMPLETED)

**Command:**
```bash
venv/Scripts/python.exe src/pipeline/process_photos.py \
  "C:\Users\kelly\OneDrive\Pictures\f1 photos\Formula 1-3-001\Formula 1" \
  --skip-vlm --skip-faces
```

**What happens:**
1. Scans directory for photos (finds 9,521 photos)
2. Extracts GP metadata from filenames
3. Generates CLIP embeddings for each photo
4. Stores in ChromaDB vector database
5. Tracks progress in SQLite database

**Time:** ~1 hour for 9,521 photos on CPU

**Resume capability:** If interrupted, restart the same command - it only processes new/unprocessed photos

**Check progress:**
```python
import sqlite3
conn = sqlite3.connect('./data/f1_photos.db')
total = conn.execute('SELECT COUNT(*) FROM photos').fetchone()[0]
embedded = conn.execute('SELECT COUNT(*) FROM photos WHERE clip_embedded_at IS NOT NULL').fetchone()[0]
print(f'{embedded}/{total} photos embedded ({embedded/total*100:.1f}%)')
```

### Layer 2: Vision LLM (NEXT)

**Step 1: Install Ollama**
```bash
# Check if installed
ollama --version

# If not: download from https://ollama.ai
```

**Step 2: Pull Vision Model**
```bash
# Download LLaVA 7B vision model (~4GB)
ollama pull llava:7b

# Verify
ollama list
```

**Step 3: Run VLM Processing**
```bash
venv/Scripts/python.exe src/pipeline/process_photos.py \
  "C:\Users\kelly\OneDrive\Pictures\f1 photos\Formula 1-3-001\Formula 1" \
  --skip-clip --skip-faces
```

**What happens:**
1. Loads each photo
2. Sends to Ollama vision model
3. Model analyzes and returns:
   - Text description (2-3 sentences)
   - Structured tags (teams, drivers, tracks, conditions, etc.)
4. Stores in database
5. Creates searchable tags

**Time:** ~13-26 hours for 9,521 photos (can run overnight)

**Resume capability:** Same as CLIP - fully resumable

**Monitor progress:**
```python
conn = sqlite3.connect('./data/f1_photos.db')
described = conn.execute('SELECT COUNT(*) FROM photos WHERE vlm_described_at IS NOT NULL').fetchone()[0]
print(f'{described}/9521 photos described')
```

### Layer 3: Face Recognition (OPTIONAL - Later)

**Requirements:**
- Install CMake: https://cmake.org/download/
- Install dlib: `pip install dlib`
- Install face_recognition: `pip install face_recognition`

**Setup reference photos:**
```
data/face_references/
├── max_verstappen/
│   ├── ref1.jpg
│   ├── ref2.jpg
│   └── ref3.jpg
├── charles_leclerc/
│   └── ...
└── your_name/
    └── ...
```

**Run face scanning:**
```bash
venv/Scripts/python.exe src/pipeline/process_photos.py \
  "C:\Users\kelly\OneDrive\Pictures\f1 photos\Formula 1-3-001\Formula 1" \
  --skip-clip --skip-vlm
```

---

## Searching Photos

### Method 1: Demo Script (Interactive)
```bash
venv/Scripts/python.exe demo_search.py
```
- Shows example searches
- Then enters interactive mode
- Type queries, get results
- Ctrl+C to exit

### Method 2: Python API
```python
from src.query.photo_search import F1PhotoSearch

search = F1PhotoSearch()

# Natural language search
results = search.search(query='red car on track', n_results=50)

# Specific Grand Prix
results = search.at_race('monaco-2024', n_results=100)

# Combined search
results = search.at_race('abu-dhabi-2021', query='podium celebration')

# Team search (requires VLM layer)
results = search.team_photos('Ferrari', query='pit stop')

# People search (requires face recognition layer)
results = search.search(people=['max_verstappen', 'charles_leclerc'])
```

### Method 3: One-line Commands
```bash
# Quick search
venv/Scripts/python.exe -c "from src.query.photo_search import F1PhotoSearch; search = F1PhotoSearch(); results = search.search(query='wet race', n_results=20); [print(f'{i}. {r[\"grand_prix\"]}: {r[\"file_path\"].split(\"/\")[-1]}') for i, r in enumerate(results, 1)]"
```

### Understanding Search Results

**Result format:**
```python
{
    'photo_id': 'abc123...',
    'file_path': '/path/to/photo.jpg',
    'grand_prix': 'monaco-2024',
    'clip_score': 0.256,           # Similarity score (higher = better match)
    'description': '...',           # VLM-generated description
    'tags': {...},                  # Structured tags
    'people': [...]                 # Detected people (if face recognition enabled)
}
```

**CLIP similarity scores:**
- `0.30+` = Excellent match
- `0.25-0.30` = Very good match
- `0.20-0.25` = Good match
- `<0.20` = Weak match

### Example Queries

**With CLIP only (current):**
- "red car on track" → Ferrari photos
- "wet race conditions" → Rain photos
- "night race" → Singapore, Abu Dhabi night races
- "podium celebration" → Podium moments
- "pit stop" → Pit lane action

**With VLM (after Layer 2):**
- `search.team_photos('Ferrari')` → All Ferrari photos
- `search.search(tags=['conditions:wet'])` → Wet races
- `search.search(tags=['session:qualifying'])` → Qualifying sessions
- "Charles Leclerc at Monaco" → More accurate results

**With Face Recognition (Layer 3):**
- `search.lestappen()` → All Max + Charles photos
- `search.search(people=['max_verstappen'])` → All Max photos
- Photos with yourself in them

---

## Understanding the Layers

### How CLIP Works
1. **Training:** CLIP was trained on millions of image-text pairs
2. **Vectors:** Converts images and text to 512-dimensional vectors
3. **Similarity:** "Red car" and Ferrari photos end up near each other in vector space
4. **Search:** Your text query is converted to a vector, then we find nearest photo vectors
5. **Zero-shot:** No F1-specific training needed - it already understands concepts

### How Vision LLM Works
1. **Input:** Photo + text prompt asking for F1-specific analysis
2. **Processing:** Model "looks" at the photo (like a human would)
3. **Output:** Generates structured JSON with tags and description
4. **Prompt:** We ask for specific F1 things (teams, drivers, tracks, conditions)
5. **Storage:** Tags become searchable in the database

### How Face Recognition Works
1. **Reference photos:** You provide 5-10 photos of each person
2. **Encoding:** System creates a "face embedding" for each person
3. **Detection:** For each photo, detect all faces
4. **Matching:** Compare detected faces to known faces
5. **Tagging:** Photos get tagged with detected people

### Why Three Layers?

Each layer catches what the others miss:
- **CLIP:** Fast, good for general concepts ("red car", "rain")
- **VLM:** Detailed, reads text, identifies specific things (car numbers, tracks)
- **Face:** Only way to reliably identify specific people

Combined, they make your collection fully searchable.

---

## Troubleshooting

### CLIP Issues

**"Model download failed"**
- Network issue - retry later
- Or manually download from Hugging Face

**"Out of memory"**
- Reduce batch_size in `clip_embedder.py`
- Close other applications

**"Process interrupted at 700 photos"**
- Normal for background tasks
- Just restart - it resumes automatically

### VLM Issues

**"Connection refused to localhost:11434"**
- Ollama not running
- Start: `ollama serve` in separate terminal

**"Model not found"**
- Pull the model: `ollama pull llava:7b`
- Check: `ollama list`

**"VLM responses not valid JSON"**
- Model sometimes adds markdown
- Code handles this - check logs for parsing errors
- Can adjust temperature in `vlm_describer.py`

**"Too slow"**
- VLM is slow by design (5-10s per photo)
- Run overnight
- Can't speed up without better hardware

### Face Recognition Issues

**"CMake not found"**
- Install CMake: https://cmake.org/download/
- Add to PATH during installation
- Or skip face recognition layer for now

**"No faces detected"**
- Helmets hide faces
- Only works on driver portraits, podium shots, paddock photos
- Not every photo will have detectable faces

**"Wrong person identified"**
- Add more reference photos (different angles/lighting)
- Adjust tolerance in `face_scanner.py`
- Drivers can look similar - face recognition isn't perfect

### Search Issues

**"No results found"**
- Check if embeddings completed: verify in database
- Try broader queries ("car" instead of "Ferrari SF-24")
- Increase n_results

**"Results don't match query"**
- CLIP is fuzzy - it finds semantic similarity
- With VLM layer, results will be more accurate
- Some queries work better than others

**"Search is slow"**
- First search loads model (slow)
- Subsequent searches are fast
- ChromaDB indexes vectors for speed

---

## Performance Stats

**Your collection:** 9,521 photos

**Processing times (measured):**
- CLIP: ~0.4s per photo = ~63 minutes total ✅ DONE
- VLM: ~5-10s per photo = ~13-26 hours total (TBD)
- Face: ~0.5s per photo = ~80 minutes total (optional)

**Storage:**
- Database: ~50 MB
- CLIP embeddings: ~200 MB
- Photos: Stay in original location (0 bytes copied)

**Hardware:**
- CPU: Works fine, slower than GPU
- GPU: Would be 3-5x faster, but network issues prevented GPU PyTorch install
- Can retry GPU setup later for VLM speed boost

---

## Next Steps

1. ✅ Complete VLM model download
2. ⏳ Run VLM processing overnight
3. 🔍 Test enhanced search with VLM tags
4. 📝 Commit progress to GitHub
5. 🎯 Optional: Add face recognition later

---

## Reference Commands

**Check database status:**
```python
import sqlite3
conn = sqlite3.connect('./data/f1_photos.db')
print(f"Total photos: {conn.execute('SELECT COUNT(*) FROM photos').fetchone()[0]}")
print(f"CLIP embedded: {conn.execute('SELECT COUNT(*) FROM photos WHERE clip_embedded_at IS NOT NULL').fetchone()[0]}")
print(f"VLM described: {conn.execute('SELECT COUNT(*) FROM photos WHERE vlm_described_at IS NOT NULL').fetchone()[0]}")
print(f"Faces scanned: {conn.execute('SELECT COUNT(*) FROM photos WHERE faces_scanned_at IS NOT NULL').fetchone()[0]}")
```

**Restart pipeline from scratch:**
```bash
# Backup database first!
cp data/f1_photos.db data/f1_photos.db.backup

# Delete database to restart
rm data/f1_photos.db

# Run pipeline again
venv/Scripts/python.exe src/pipeline/process_photos.py "..." --skip-vlm --skip-faces
```

**Push to GitHub:**
```bash
git add .
git commit -m "Update progress"
git push
```

---

*Last updated: 2026-02-11*
*System: F1 Photo Intelligence v1.0*
