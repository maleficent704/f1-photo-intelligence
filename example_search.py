#!/usr/bin/env python3
"""
Example usage of F1 Photo Intelligence search system.

Make sure you've run the processing pipeline first:
    python src/pipeline/process_photos.py ~/Photos/F1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query.photo_search import F1PhotoSearch

def main():
    print("F1 Photo Intelligence - Search Examples\n")

    # Initialize search
    search = F1PhotoSearch()

    # Example 1: Natural language search
    print("1. Searching for 'rainy race start'...")
    results = search.search(query='rainy race start', n_results=5)
    print(f"   Found {len(results)} photos")
    for r in results[:3]:
        print(f"   - {r['file_path']} (score: {r['clip_score']:.3f})")

    # Example 2: Lestappen content
    print("\n2. Finding Lestappen moments...")
    results = search.lestappen(n_results=5)
    print(f"   Found {len(results)} photos with Max and Charles")
    for r in results[:3]:
        print(f"   - {r['file_path']}")

    # Example 3: Team photos
    print("\n3. Ferrari photos...")
    results = search.team_photos('Ferrari', n_results=5)
    print(f"   Found {len(results)} Ferrari photos")

    # Example 4: Specific race
    print("\n4. Photos from Monaco 2024...")
    results = search.at_race('monaco-2024', n_results=5)
    print(f"   Found {len(results)} photos from Monaco")

    # Example 5: Combined search
    print("\n5. Charles celebrating at Monza...")
    results = search.search(
        query='celebrating',
        people=['charles_leclerc'],
        grand_prix='monza-2024',
        n_results=5
    )
    print(f"   Found {len(results)} matching photos")

    # Example 6: Tag-based search
    print("\n6. Wet race conditions...")
    results = search.search(
        tags=['conditions:wet', 'session:race'],
        n_results=5
    )
    print(f"   Found {len(results)} wet race photos")

    print("\n✓ Search examples complete!")

if __name__ == '__main__':
    main()
