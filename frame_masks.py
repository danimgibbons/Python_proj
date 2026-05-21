import cv2
import numpy as np
from pathlib import Path

# --- INPUTS ---
video_path = "/Users/danimgibbons/Desktop/PhD/skater_proj/vids/chunks_Dani/video3/cleaned/big1_small1_cleaned.mp4"
reference_frame_path = "/Users/danimgibbons/Desktop/PhD/skater_proj/vids/frames/c3_empty.png"
background_mask_path = "/Users/danimgibbons/Desktop/PhD/skater_proj/mask_c3_cones.png"

output_root = "masks"

# --- SETUP ---
video_path = Path(video_path)
output_dir = Path(output_root) / video_path.stem

output_dir.mkdir(parents=True, exist_ok=True)

ref = cv2.imread(str(reference_frame_path), cv2.IMREAD_UNCHANGED)
background_mask = cv2.imread(str(background_mask_path), cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(str(video_path))

frame_idx = 0
accumulator = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- float ---
    ref_f = ref.astype("float32")
    frame_f = frame.astype("float32")

    # --- difference ---
    diff = cv2.absdiff(frame_f, ref_f)
    diff = cv2.GaussianBlur(diff, (21,21), 0)

    # --- convert to grayscale BEFORE threshold ---
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # --- threshold ---
    _, skater_mask = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)

    # --- STEP 1: merge nearby blobs ---
    kernel = np.ones((3, 3), np.uint8)  # adjust size depending on how far blobs are
    skater_mask = cv2.morphologyEx(skater_mask, cv2.MORPH_CLOSE, kernel)

    # --- STEP 2: keep only largest blob ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skater_mask, connectivity=8)

    if num_labels > 1:
        # skip background (label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        largest_blob = np.zeros_like(skater_mask)
        largest_blob[labels == largest_label] = 255

        skater_mask = largest_blob

    # --- apply mask (set skater pixels to white) ---
    final_mask = background_mask.copy()
    final_mask[skater_mask == 255] = [0, 0, 0, 0]

    # --- save ---
    out_path = output_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), final_mask)

    frame_idx += 1
    print(frame_idx)

print(f"Done. Final masks saved to: {output_dir}")