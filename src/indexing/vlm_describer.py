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

        prompt = """Analyze this Formula 1 photograph in detail. You are an expert F1 photographer analyzing race photos.

TEAM IDENTIFICATION GUIDE (2020-2024):
- Ferrari: Red/burgundy cars
- Red Bull Racing: Dark blue/navy cars (often with red accents)
- Mercedes: Silver or black cars (black in 2020-2021, silver after)
- McLaren: Orange/papaya cars (bright orange)
- Aston Martin: Green cars (British racing green)
- Alpine: Blue and pink cars
- Williams: Blue cars
- Alfa Romeo/Sauber: Red and white cars
- Haas: White and red or grey cars
- AlphaTauri/RB: White and blue cars

TRACK IDENTIFICATION:
- Monaco: Tight barriers, Armco, street circuit
- Spa: Forested background, elevation changes
- Silverstone: Wide open spaces, British flags
- Monza: Historic buildings, long straights
- Singapore: Night race, Marina Bay, city lights
- Abu Dhabi: Yas Marina, twilight/night, modern architecture

1. DESCRIPTION: Provide a detailed 2-3 sentence description focusing on:
   - What team/driver is visible (identify by car color and livery)
   - What's happening in the photo
   - Any identifiable track features or context

2. TAGS: Extract structured information:

- teams: Identify F1 teams by car color and livery. Be specific (e.g., "Ferrari", "Red Bull Racing", "Aston Martin"). If you see green cars, it's likely Aston Martin. Orange cars are McLaren. Red cars are Ferrari.

- drivers: Any identifiable drivers by car numbers, helmet designs, or names visible

- track: Which circuit based on distinctive features, scenery, barriers, or architecture

- session: race, qualifying, practice, podium, pit_lane, paddock, garage, or team_photo

- conditions: Weather and track (dry, wet, overcast, sunny, night, sunset, twilight)

- action: on-track, pit_stop, celebration, crash, team_photo, or portrait

- car_numbers: Any visible racing numbers on cars (1-99)

- tire_compound: soft (red), medium (yellow), hard (white), intermediate (green), wet (blue)

- objects: trophy, champagne, flag, helmet, steering_wheel, pit_board

- people_count: Approximate count of people clearly visible (0-100)

- composition: close-up, wide_shot, aerial, portrait, action_shot, or team_photo

IMPORTANT:
- Look carefully at car colors to identify teams
- If you see team uniforms/clothing, note the colors
- Car numbers are very helpful for driver identification
- Be specific with team names (not just "Formula 1")

Respond ONLY with valid JSON, no other text:
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

        # Check response status
        response.raise_for_status()

        # Parse response
        response_data = response.json()
        if 'response' not in response_data:
            logging.error(f"No 'response' key in Ollama output. Got: {response_data.keys()}")
            raise KeyError(f"Ollama response missing 'response' key. Response: {response_data}")

        result = response_data['response']

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
