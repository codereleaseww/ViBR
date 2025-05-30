import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse
import sys

def compute_ssim(imageA, imageB, threshold=0.95):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    print(f"SSIM Score: {score:.4f}")
    return score > threshold

def compute_abs_diff(imageA, imageB, threshold=10):
    diff = cv2.absdiff(imageA, imageB)
    mean_diff = np.mean(diff)
    print(f"Absolute Difference (mean): {mean_diff:.2f}")
    return mean_diff < threshold

def compute_sift_matches(imageA, imageB, threshold=0.25):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayA, None)
    kp2, des2 = sift.detectAndCompute(grayB, None)

    if des1 is None or des2 is None:
        print("SIFT descriptors not found.")
        return False

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    similarity = len(good) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0

    print(f"SIFT Similarity: {similarity:.4f} ({len(good)} good matches)")
    return similarity > threshold

def compare_images(img_path1, img_path2, method):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        sys.exit(1)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    method = method.upper()

    if method == "SSIM":
        result = compute_ssim(img1, img2)
    elif method == "ABS":
        result = compute_abs_diff(img1, img2)
    elif method == "SIFT":
        result = compute_sift_matches(img1, img2)
    else:
        print("Error: Method must be one of 'SSIM', 'ABS', or 'SIFT'.")
        sys.exit(1)

    print(f"\nClassified as Similar: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI state comparison.")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument("method", help="Comparison method: SSIM, ABS, or SIFT")
    args = parser.parse_args()

    compare_images(args.image1, args.image2, args.method)
