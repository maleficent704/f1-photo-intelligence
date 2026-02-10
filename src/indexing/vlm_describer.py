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
                    "temperature": 0.1,
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
                "description": result,
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
                    elif values:
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
