"""
CAMS – Train CLI
Run this once to build the LBPH model from your target profile images.

Usage
-----
  python train.py

Place the target person's photos (any quantity, any angle) inside the
`target_profile/` directory at the project root.

For negative examples (other people the system should NOT match), create a
`target_profile/negatives/` sub-directory and add photos there.

After training, the model is saved to models/lbph_model.xml and will be
loaded automatically on every subsequent run.
"""
import logging
import sys
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

from config import TARGET_DIR, LBPH_MODEL_PATH
from recognition import FaceRecognitionEngine

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    photos = [f for f in os.listdir(TARGET_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    if not photos:
        print(f"\n⚠  No target images found in '{TARGET_DIR}'.")
        print("   Add at least one clear, well-lit face photo of the target person.")
        print("   Supported formats: .jpg .jpeg .png .bmp\n")
        sys.exit(1)

    print(f"\n📸  Found {len(photos)} image(s) in '{TARGET_DIR}'.")
    print("    Training LBPH face recogniser…\n")

    engine = FaceRecognitionEngine()
    ok = engine.train_from_directory(TARGET_DIR)
    if ok:
        print(f"\n✅  Model saved → {LBPH_MODEL_PATH}")
        print("    You can now run: python main.py\n")
    else:
        print("\n❌  Training failed — no usable face crops found.")
        print("    Make sure the photos are well-lit and the face is clearly visible.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
