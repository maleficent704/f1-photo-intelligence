import sqlite3
import json
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
                    candidate_ids &= person_photos
                else:
                    candidate_ids |= person_photos

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
                    n_results=min(n_results * 3, len(candidate_ids)),
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
