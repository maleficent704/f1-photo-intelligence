# Face Reference Photos

This directory stores reference photos for face recognition.

## Setup

Create a subdirectory for each person you want to recognize, using a consistent ID (e.g., lowercase with underscores):

```
face_references/
├── max_verstappen/
│   ├── ref_1.jpg
│   ├── ref_2.jpg
│   └── ref_3.jpg
├── charles_leclerc/
│   └── ...
├── lando_norris/
│   └── ...
└── your_name/
    └── ...
```

## Tips for Good Reference Photos

- **5-10 photos per person** is ideal
- Include different angles (front, 3/4 profile)
- Include different lighting conditions
- Include with and without accessories (sunglasses, hats, helmets)
- Clear face visibility is more important than high resolution
- Front-facing or 3/4 angle shots work best

## Examples

**F1 Drivers:**
- `max_verstappen/` - Max Verstappen
- `charles_leclerc/` - Charles Leclerc
- `lando_norris/` - Lando Norris
- `lewis_hamilton/` - Lewis Hamilton
- `george_russell/` - George Russell
- etc.

**Team Personnel:**
- `christian_horner/` - Red Bull Team Principal
- `fred_vasseur/` - Ferrari Team Principal
- etc.

**Personal:**
- `your_name/` - Yourself in photos
- `family_member/` - Family at races
- etc.

## Registering People

After adding reference photos, register the person in the database:

```python
from src.indexing.face_scanner import FaceScanner

scanner = FaceScanner()
scanner.add_reference_person(
    person_id='max_verstappen',
    name='Max Verstappen',
    category='driver',
    team='Red Bull Racing'
)
```

Or add multiple people at once in a script.
