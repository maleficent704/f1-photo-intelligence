import argparse
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.indexing.clip_embedder import CLIPEmbedder
from src.indexing.vlm_describer import VLMDescriber
# face_scanner imported conditionally if needed

def setup_logging():
    """Configure logging to both file and console."""
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'photo_pipeline.log'),
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

    setup_logging()
    logger = logging.getLogger(__name__)

    db_path = './data/f1_photos.db'

    logger.info("=" * 60)
    logger.info("F1 Photo Intelligence Pipeline")
    logger.info("=" * 60)

    # Step 1: Discover photos
    logger.info("Step 1: Scanning for new photos...")
    embedder = CLIPEmbedder(db_path=db_path)
    new_photos = embedder.scan_photos(photo_dir)
    logger.info(f"Found {new_photos} photos")

    # Step 2: CLIP embeddings (fast)
    if not skip_clip:
        logger.info("\nStep 2: Generating CLIP embeddings...")
        embedded = embedder.embed_batch()
        logger.info(f"Embedded {embedded} photos")
    else:
        logger.info("\nStep 2: Skipping CLIP embeddings (--skip-clip)")

    # Step 3: Face recognition (medium speed)
    if not skip_faces:
        logger.info("\nStep 3: Scanning for faces...")
        from src.indexing.face_scanner import FaceScanner
        scanner = FaceScanner(db_path=db_path)
        scanned = scanner.process_batch()
        logger.info(f"Scanned {scanned} photos for faces")
    else:
        logger.info("\nStep 3: Skipping face recognition (--skip-faces)")

    # Step 4: VLM descriptions (slow - do last)
    if not skip_vlm:
        logger.info("\nStep 4: Generating VLM descriptions...")
        if vlm_limit:
            logger.info(f"Processing limited to {vlm_limit} photos")
        describer = VLMDescriber(db_path=db_path)
        described = describer.process_batch(limit=vlm_limit)
        logger.info(f"Described {described} photos")
    else:
        logger.info("\nStep 4: Skipping VLM descriptions (--skip-vlm)")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='F1 Photo Intelligence Pipeline - AI-powered photo tagging and search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on all photos
  python process_photos.py ~/Photos/F1

  # Just CLIP embeddings (fast, get searching quickly)
  python process_photos.py ~/Photos/F1 --skip-vlm --skip-faces

  # Test VLM on 100 photos
  python process_photos.py ~/Photos/F1 --skip-clip --skip-faces --vlm-limit 100

  # Just face recognition
  python process_photos.py ~/Photos/F1 --skip-clip --skip-vlm
        """
    )
    parser.add_argument('photo_dir', help='Path to F1 photos directory')
    parser.add_argument('--skip-clip', action='store_true',
                        help='Skip CLIP embedding generation')
    parser.add_argument('--skip-vlm', action='store_true',
                        help='Skip VLM description generation (slow)')
    parser.add_argument('--skip-faces', action='store_true',
                        help='Skip face recognition scanning')
    parser.add_argument('--vlm-limit', type=int,
                        help='Limit VLM processing to N photos (for testing)')

    args = parser.parse_args()

    # Validate photo directory exists
    photo_path = Path(args.photo_dir)
    if not photo_path.exists():
        print(f"Error: Photo directory does not exist: {args.photo_dir}")
        sys.exit(1)

    run_pipeline(
        args.photo_dir,
        skip_clip=args.skip_clip,
        skip_vlm=args.skip_vlm,
        skip_faces=args.skip_faces,
        vlm_limit=args.vlm_limit
    )

if __name__ == '__main__':
    main()
