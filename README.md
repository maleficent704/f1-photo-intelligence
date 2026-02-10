# F1 Photo Intelligence

AI-powered photo tagging and search system for F1 photo collections. Transform folders of race photos into a richly searchable archive using CLIP embeddings, vision LLMs, and face recognition.

## Features

**Three-layer AI system:**

1. **CLIP Embeddings** - Natural language photo search ("wet race starts", "podium celebrations")
2. **Vision LLM Descriptions** - Structured F1-specific tagging (teams, drivers, tracks, conditions)
3. **Face Recognition** - Find photos containing specific people (drivers, team principals, yourself)

**Combined search examples:**
- All photos with Max and Charles together (Lestappen!)
- Ferrari pit stops at Monza
- Rainy qualifying sessions
- Photos of you at Silverstone

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/maleficent704/f1-photo-intelligence.git
cd f1-photo-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: face_recognition requires cmake
# Mac: brew install cmake
# Linux: sudo apt install cmake
```

### Photo Directory Structure

Organize your F1 photos by Grand Prix:

```
~/Photos/F1/
├── bahrain-2024/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
├── monaco-2024/
│   └── ...
└── silverstone-2024/
    └── ...
```

### Run the Pipeline

**Get searching fast (CLIP only - ~3-4 hours for 12k photos):**
```bash
python src/pipeline/process_photos.py ~/Photos/F1 --skip-vlm --skip-faces
```

**Add face recognition (~2 hours):**
```bash
python src/pipeline/process_photos.py ~/Photos/F1 --skip-clip --skip-vlm
```

**Full pipeline (runs overnight):**
```bash
python src/pipeline/process_photos.py ~/Photos/F1
```

### Face Recognition Setup

Create reference photos for people you want to recognize:

```
data/face_references/
├── max_verstappen/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── charles_leclerc/
│   └── ...
└── your_name/
    └── ...
```

5-10 reference photos per person is ideal. Include different angles and lighting conditions.

### VLM Setup

Pull a vision model in Ollama (if not already installed):

```bash
# Recommended model
ollama pull qwen2-vl:7b

# Alternatives
ollama pull llava:13b
ollama pull minicpm-v:8b
```

## Usage

### Python API

```python
from src.query.photo_search import F1PhotoSearch

search = F1PhotoSearch()

# Natural language search
results = search.search(query='rainy race start')

# Search by people
results = search.lestappen()  # Max + Charles
results = search.search(people=['max_verstappen'])

# Combined search
results = search.search(
    query='celebrating',
    people=['charles_leclerc'],
    grand_prix='monza-2024'
)

# Team searches
results = search.team_photos('Ferrari')
results = search.team_photos('Ferrari', query='pit stop')

# Tag-based filtering
results = search.search(tags=['conditions:wet', 'session:race'])
```

### Results Format

```python
[
    {
        'photo_id': 'abc123...',
        'file_path': '/path/to/photo.jpg',
        'grand_prix': 'monaco-2024',
        'clip_score': 0.85,
        'description': 'Ferrari on Monaco street circuit...',
        'tags': {
            'teams': ['Ferrari'],
            'track': 'Monaco',
            'session': 'race',
            ...
        },
        'people': [
            {'person_id': 'charles_leclerc', 'confidence': 0.92}
        ]
    },
    ...
]
```

## Architecture

```
┌─────────────────────┐
│   Photo Source      │  ~/Photos/F1/{gp}/*.jpg
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Processing Pipeline (RTX 4080)     │
│                                     │
│  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │ CLIP │  │ VLM  │  │ Face │     │
│  │  1s  │  │ 5-10s│  │ 0.5s │     │
│  └──┬───┘  └──┬───┘  └──┬───┘     │
│     └─────────┼─────────┘          │
└───────────────┼────────────────────┘
                │
                ▼
┌──────────────────────────────┐
│  SQLite + ChromaDB Storage   │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  Query Interface             │
│  - Natural language search   │
│  - Tag filtering             │
│  - Face search               │
│  - Combined queries          │
└──────────────────────────────┘
```

## Hardware Requirements

**GPU:** NVIDIA GPU with 16GB VRAM recommended (RTX 4080/4090, A4000, etc.)

**Processing time for 12k photos:**
- CLIP: ~3-4 hours
- VLM: ~17-33 hours (can be interrupted/resumed)
- Face recognition: ~1.5-2 hours

**Storage:** ~500MB for metadata and embeddings (photos remain in original location)

## Project Structure

```
f1-photo-intelligence/
├── src/
│   ├── indexing/
│   │   ├── clip_embedder.py      # CLIP embedding generation
│   │   ├── vlm_describer.py      # Vision LLM descriptions
│   │   └── face_scanner.py       # Face detection/recognition
│   ├── query/
│   │   └── photo_search.py       # Combined search interface
│   └── pipeline/
│       └── process_photos.py     # Pipeline runner
├── data/                         # Generated (not in git)
│   ├── f1_photos.db             # SQLite metadata
│   ├── chroma_db/               # CLIP embeddings
│   └── face_references/         # Reference photos
├── logs/                        # Processing logs
└── requirements.txt
```

## Model Selection

### CLIP Models

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| `clip-ViT-B-32` | ~1GB | Fast | Good |
| `clip-ViT-L-14` | ~2GB | Medium | Better (recommended) |
| `clip-ViT-H-14` | ~3GB | Slow | Best |

### Vision LLM Models

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| `llava:7b` | ~5GB | ~3s/img | Good |
| `qwen2-vl:7b` | ~5GB | ~4s/img | Very good (recommended) |
| `llava:13b` | ~9GB | ~6s/img | Better |

## Future Enhancements

- Web UI for browsing and searching
- Auto-processing of new photos
- Smart album generation
- Duplicate detection
- Enhanced models as they improve

## License

MIT

## Acknowledgments

Built using:
- [CLIP](https://github.com/openai/CLIP) via sentence-transformers
- [ChromaDB](https://www.trychroma.com/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Ollama](https://ollama.ai/) for local vision LLMs
