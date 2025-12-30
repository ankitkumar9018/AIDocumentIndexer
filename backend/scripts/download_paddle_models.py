"""
PaddleOCR Model Downloader
===========================

Pre-downloads PaddleOCR models for specified languages to avoid runtime delays.
Models are cached locally for faster OCR processing.
"""

import os
import sys

# Set environment variables before importing PaddleOCR
# Use project directory instead of system directory for portability
PADDLE_HOME = os.path.join(os.getcwd(), "data", "paddle_models")
PADDLE_HUB = os.path.join(PADDLE_HOME, "official_models")

os.environ['PADDLEX_HOME'] = PADDLE_HOME
os.environ['PADDLE_HUB_HOME'] = PADDLE_HUB
os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'HF'  # Use HuggingFace mirror

print(f"PaddleOCR Model Downloader")
print(f"=" * 50)
print(f"Cache directory: {PADDLE_HOME}")
print(f"Model source: HuggingFace")
print(f"Note: Models stored in project directory for portability")
print()

# Create directories
os.makedirs(PADDLE_HOME, exist_ok=True)
os.makedirs(PADDLE_HUB, exist_ok=True)

try:
    from paddleocr import PaddleOCR

    # Languages to download (German and English)
    languages = ['de', 'en']

    for lang in languages:
        print(f"Downloading {lang.upper()} models...")
        print(f"  - Detection model")
        print(f"  - Recognition model")
        print(f"  - Classification model")

        try:
            # Initialize PaddleOCR to trigger model download
            # Note: use_textline_orientation replaces deprecated use_angle_cls
            # show_log parameter removed in newer versions
            # Disable doc preprocessing to avoid UVDoc model issues
            ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=lang,
            )
            print(f"✓ {lang.upper()} models downloaded successfully")
            print()
        except Exception as e:
            print(f"✗ Failed to download {lang.upper()} models: {e}")
            print()

    print("=" * 50)
    print("Model download complete!")
    print(f"Models cached at: {PADDLE_HOME}")
    sys.exit(0)

except ImportError as e:
    print(f"✗ Error: PaddleOCR not installed")
    print(f"  Install with: pip install paddleocr paddlepaddle")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
