#!/usr/bin/env python3
"""
Quick search demo for F1 Photo Intelligence.

Try different search queries to see how CLIP semantic search works!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query.photo_search import F1PhotoSearch

def main():
    print("=" * 60)
    print("F1 Photo Intelligence - Search Demo")
    print("=" * 60)

    search = F1PhotoSearch()

    # Demo searches
    demos = [
        {
            "title": "Natural Language Search",
            "searches": [
                ("red car on track", 5),
                ("podium celebration", 5),
                ("wet race conditions", 5),
                ("pit stop", 5),
            ]
        },
        {
            "title": "Grand Prix Specific",
            "searches": [
                ("abu-dhabi-2021", 5),
            ]
        },
        {
            "title": "Combined Searches",
            "searches": [
                ("at_race:abu-dhabi-2021 + celebrating", 3),
            ]
        }
    ]

    for demo_group in demos:
        print(f"\n{demo_group['title']}")
        print("-" * 60)

        for query_info in demo_group['searches']:
            if isinstance(query_info, tuple):
                query, n = query_info
            else:
                query = query_info
                n = 5

            print(f"\nQuery: '{query}'")

            # Handle different query types
            if query.startswith("at_race:"):
                # Format: "at_race:gp_name + query"
                parts = query.split(" + ")
                gp = parts[0].replace("at_race:", "")
                q = parts[1] if len(parts) > 1 else None
                results = search.at_race(gp, query=q, n_results=n)
            else:
                results = search.search(query=query, n_results=n)

            print(f"Found {len(results)} photos:")
            for i, r in enumerate(results[:n], 1):
                filename = Path(r['file_path']).name
                score = r.get('clip_score', 0)
                gp = r.get('grand_prix', 'unknown')
                print(f"  {i}. {filename}")
                print(f"     GP: {gp}, Score: {score:.3f}")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Search Mode")
    print("=" * 60)
    print("Try your own searches! (Ctrl+C to exit)")
    print()

    while True:
        try:
            query = input("Search: ").strip()
            if not query:
                continue

            results = search.search(query=query, n_results=10)
            print(f"\nFound {len(results)} photos:")

            for i, r in enumerate(results, 1):
                filename = Path(r['file_path']).name
                score = r.get('clip_score', 0)
                gp = r.get('grand_prix', 'unknown')
                print(f"  {i}. {filename}")
                print(f"     GP: {gp}, Score: {score:.3f}")
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
