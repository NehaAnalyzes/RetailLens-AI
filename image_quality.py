"""
image_quality.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retail Product Image Quality Checker
Checks: Resolution, Blur, Brightness
Output: PASS/FAIL + Quality Score (0-100)

Usage:
  Single image:  python image_quality.py --image product.jpg
  Whole folder:  python image_quality.py --folder val/ELECTRONICS
  Full val set:  python image_quality.py --valdir val
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# THRESHOLDS  (tweak if needed)
# ─────────────────────────────────────────────
MIN_WIDTH       = 100    # pixels
MIN_HEIGHT      = 100    # pixels
BLUR_THRESHOLD  = 80.0   # Laplacian variance — below = blurry
BRIGHT_MIN      = 40     # mean brightness — below = too dark
BRIGHT_MAX      = 230    # mean brightness — above = overexposed


# ─────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────
def check_resolution(img):
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        score = max(0, int(min(w, h) / max(MIN_WIDTH, MIN_HEIGHT) * 100))
        return False, score, f"Too small ({w}x{h}px, min {MIN_WIDTH}x{MIN_HEIGHT})"
    score = min(100, int((min(w, h) / 224) * 100))
    return True, score, f"OK ({w}x{h}px)"


def check_blur(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    score    = min(100, int((variance / (BLUR_THRESHOLD * 2)) * 100))
    if variance < BLUR_THRESHOLD:
        return False, score, f"Blurry (sharpness={variance:.1f}, min={BLUR_THRESHOLD})"
    return True, score, f"Sharp (sharpness={variance:.1f})"


def check_brightness(img):
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_bright = gray.mean()
    if mean_bright < BRIGHT_MIN:
        score = max(0, int((mean_bright / BRIGHT_MIN) * 50))
        return False, score, f"Too dark (brightness={mean_bright:.1f}, min={BRIGHT_MIN})"
    if mean_bright > BRIGHT_MAX:
        over  = mean_bright - BRIGHT_MAX
        score = max(0, int(50 - (over / (255 - BRIGHT_MAX)) * 50))
        return False, score, f"Overexposed (brightness={mean_bright:.1f}, max={BRIGHT_MAX})"
    mid   = (BRIGHT_MIN + BRIGHT_MAX) / 2
    dist  = abs(mean_bright - mid) / ((BRIGHT_MAX - BRIGHT_MIN) / 2)
    score = int((1 - dist * 0.3) * 100)
    return True, min(100, score), f"OK (brightness={mean_bright:.1f})"


# ─────────────────────────────────────────────
# MAIN CHECKER
# ─────────────────────────────────────────────
def check_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "file":    str(image_path),
            "result":  "FAIL",
            "score":   0,
            "issues":  ["Cannot read image file"],
            "details": {}
        }

    res_pass,    res_score,    res_msg    = check_resolution(img)
    blur_pass,   blur_score,   blur_msg   = check_blur(img)
    bright_pass, bright_score, bright_msg = check_brightness(img)

    # Weighted overall score
    overall_score = int(
        res_score    * 0.30 +
        blur_score   * 0.40 +
        bright_score * 0.30
    )

    issues = []
    if not res_pass:    issues.append(res_msg)
    if not blur_pass:   issues.append(blur_msg)
    if not bright_pass: issues.append(bright_msg)

    return {
        "file":    str(image_path),
        "result":  "PASS" if not issues else "FAIL",
        "score":   overall_score,
        "issues":  issues,
        "details": {
            "resolution": {"pass": res_pass,    "score": res_score,    "msg": res_msg},
            "blur":       {"pass": blur_pass,   "score": blur_score,   "msg": blur_msg},
            "brightness": {"pass": bright_pass, "score": bright_score, "msg": bright_msg},
        }
    }


def print_result(r, verbose=True):
    icon = "✅" if r["result"] == "PASS" else "❌"
    name = Path(r["file"]).name
    print(f"{icon} [{r['result']}] Score: {r['score']:3d}/100  |  {name}")
    if verbose and r["issues"]:
        for issue in r["issues"]:
            print(f"      ⚠️  {issue}")


def check_folder(folder_path, verbose=True):
    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images  = [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {folder_path}")
        return

    print(f"\n📁 Checking {len(images)} images in: {folder_path}")
    print("─" * 60)

    results   = [check_image(img) for img in sorted(images)]
    passed    = [r for r in results if r["result"] == "PASS"]
    failed    = [r for r in results if r["result"] == "FAIL"]
    avg_score = int(np.mean([r["score"] for r in results]))

    for r in results:
        print_result(r, verbose=verbose)

    print("─" * 60)
    print(f"📊 SUMMARY")
    print(f"   Total images : {len(results)}")
    print(f"   ✅ PASS       : {len(passed)}  ({100*len(passed)//len(results)}%)")
    print(f"   ❌ FAIL       : {len(failed)}  ({100*len(failed)//len(results)}%)")
    print(f"   Average Score: {avg_score}/100")

    if failed:
        print(f"\n⚠️  COMMON ISSUES:")
        issue_counts = {}
        for r in failed:
            for issue in r["issues"]:
                key = issue.split("(")[0].strip()
                issue_counts[key] = issue_counts.get(key, 0) + 1
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"   {count:3d}x  {issue}")


def check_valdir(val_dir):
    val_path = Path(val_dir)
    classes  = sorted([d for d in val_path.iterdir() if d.is_dir()])
    if not classes:
        print(f"No class folders found in {val_dir}")
        return

    print(f"\n🗂️  Checking entire val set: {val_dir}")
    print("=" * 65)
    print(f"  {'CLASS':<30} | {'IMGS':>4} | {'PASS':>9} | {'AVG SCORE':>9}")
    print("=" * 65)

    all_results = []
    for cls_dir in classes:
        exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images  = [p for p in cls_dir.rglob("*") if p.suffix.lower() in exts]
        results = [check_image(img) for img in images]
        all_results.extend(results)

        passed    = sum(1 for r in results if r["result"] == "PASS")
        avg_score = int(np.mean([r["score"] for r in results])) if results else 0
        pct       = 100 * passed // max(len(results), 1)
        print(f"  {cls_dir.name:<30} | {len(images):4d} | "
              f"{passed:4d} ({pct:3d}%) | {avg_score:6d}/100")

    print("=" * 65)
    total     = len(all_results)
    passed    = sum(1 for r in all_results if r["result"] == "PASS")
    avg_score = int(np.mean([r["score"] for r in all_results])) if all_results else 0
    pct       = 100 * passed // max(total, 1)
    print(f"  {'TOTAL':<30} | {total:4d} | "
          f"{passed:4d} ({pct:3d}%) | {avg_score:6d}/100")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce Image Quality Checker")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--valdir", type=str, help="Path to val/ directory (checks all classes)")
    parser.add_argument("--quiet",  action="store_true", help="Hide per-image issue details")
    args = parser.parse_args()

    if args.image:
        r = check_image(args.image)
        print_result(r, verbose=True)
        print(f"\n📋 Detailed Breakdown:")
        for check, info in r["details"].items():
            icon = "✅" if info["pass"] else "❌"
            print(f"   {icon} {check:<12} Score: {info['score']:3d}/100  |  {info['msg']}")

    elif args.folder:
        check_folder(args.folder, verbose=not args.quiet)

    elif args.valdir:
        check_valdir(args.valdir)

    else:
        # Default: run on val directory
        default_val = os.path.join(
            r"C:\PLACEMENT CHALLENGE\PROJECTS\COMPUTER VISION\archive (1)\ECOMMERCE_PRODUCT_IMAGES",
            "val"
        )
        print("No arguments provided. Running on default val directory...")
        check_valdir(default_val)
        print("\n💡 Tip — run with arguments for more control:")
        print("   python image_quality.py --image product.jpg")
        print("   python image_quality.py --folder val/ELECTRONICS")
        print("   python image_quality.py --valdir val")