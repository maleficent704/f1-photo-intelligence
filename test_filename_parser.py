#!/usr/bin/env python3
"""Test the GP filename parser with real filenames."""

import re

def parse_gp_from_filename(filename: str) -> dict:
    """
    Extract GP metadata from filename.

    Expected format: "YYYY GP_Name GP - Additional Info - Subject.jpg"
    Example: "2021 Abu Dhabi GP - Post Season Testing - Carlos Sainz.jpg"

    Returns dict with 'year' and 'grand_prix' keys.
    """
    # Pattern: YYYY followed by GP name ending with "GP"
    match = re.match(r'^(\d{4})\s+(.+?\s+GP)', filename)

    if match:
        year = match.group(1)
        gp_name = match.group(2).strip()
        return {
            'year': year,
            'grand_prix': f"{gp_name.replace(' GP', '').lower().replace(' ', '-')}-{year}",
            'gp_raw': gp_name
        }

    # Fallback: no parseable GP info
    return {
        'year': None,
        'grand_prix': 'unknown',
        'gp_raw': None
    }

# Test with the example filename
test_files = [
    "2021 Abu Dhabi GP - Post Season Testing - Carlos Sainz.jpg",
    "2024 Monaco GP - Qualifying - Max Verstappen.jpg",
    "2023 British GP - Race - Lewis Hamilton.jpg",
    "random_photo.jpg",  # Should fall back to 'unknown'
]

print("Testing GP filename parser:\n")
for filename in test_files:
    result = parse_gp_from_filename(filename)
    print(f"File: {filename}")
    print(f"  Year: {result['year']}")
    print(f"  GP: {result['grand_prix']}")
    print(f"  Raw GP: {result['gp_raw']}")
    print()
