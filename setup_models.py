"""
CAMS – Model Setup
Downloads the OpenCV DNN face detector model files from the official
OpenCV GitHub repository. Run this ONCE before first launch.

Usage:  python setup_models.py
"""
import os
import sys
import urllib.request
import hashlib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DOWNLOADS = [
    {
        "name":     "deploy.prototxt",
        "url":      "https://raw.githubusercontent.com/opencv/opencv/master/"
                    "samples/dnn/face_detector/deploy.prototxt",
        "dest":     os.path.join(MODEL_DIR, "deploy.prototxt"),
        "size_kb":  "~3 KB",
    },
    {
        "name":     "res10_300x300_ssd_iter_140000.caffemodel",
        "url":      "https://github.com/opencv/opencv_3rdparty/raw/"
                    "dnn_samples_face_detector_20170830/"
                    "res10_300x300_ssd_iter_140000.caffemodel",
        "dest":     os.path.join(MODEL_DIR,
                                 "res10_300x300_ssd_iter_140000.caffemodel"),
        "size_kb":  "~10 MB",
    },
]


def _progress(count, block, total):
    done = int(50 * count * block / max(total, 1))
    pct  = min(100, count * block * 100 // max(total, 1))
    bar  = "█" * done + "░" * (50 - done)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\n📦  CAMS Model Setup\n" + "─" * 50)
    all_ok = True
    for item in DOWNLOADS:
        dest = item["dest"]
        if os.path.exists(dest):
            size = os.path.getsize(dest)
            print(f"  ✅  {item['name']}  ({size // 1024} KB) – already present")
            continue
        print(f"\n  ⬇   Downloading {item['name']}  ({item['size_kb']}) …")
        try:
            urllib.request.urlretrieve(item["url"], dest, reporthook=_progress)
            print()   # newline after progress bar
            size = os.path.getsize(dest)
            print(f"  ✅  Saved → {dest}  ({size // 1024} KB)")
        except Exception as exc:
            print(f"\n  ❌  Failed: {exc}")
            all_ok = False

    print("\n" + "─" * 50)
    if all_ok:
        print("  All models ready.  Next steps:")
        print("  1.  Place target photo(s) in ./target_profile/")
        print("  2.  python train.py")
        print("  3.  python main.py\n")
    else:
        print("  Some downloads failed. Check your internet connection.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
