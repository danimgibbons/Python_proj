import cv2
import numpy as np
import subprocess
import os

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "/Users/danimgibbons/Desktop/PhD/skater_proj/vids/chunks_Dani/video3/cleaned/big1_small1_cleaned.mp4"
MASK_PATH = "/Users/danimgibbons/Desktop/PhD/skater_proj/masks/big1_small1_cleaned/"
OUTPUT_PATH = "/Users/danimgibbons/Desktop/PhD/skater_proj/vids/chunks_Dani/output_c3.mp4"

# Define colors for each mask color (BGR)
COLOR_MAP = {
    (255, 0, 0): ((255, 0, 0), 0.2),
    (0, 0, 255): ((0, 0, 255), 0.2),
    (100, 0, 0): ((255, 0, 0), 0.4),
    (0, 100, 0): ((0, 255, 0), 0.4),
    (0, 0, 100): ((0, 0, 255), 0.4),
    (0, 0, 0): ((0, 255, 255), 0.4),
}

tolerance = 10

# -----------------------------
# PRECOMPUTE COLOR ARRAYS
# -----------------------------
colors = np.array(list(COLOR_MAP.keys()), dtype=np.int16)          # (N,3)
overlays = np.array([v[0] for v in COLOR_MAP.values()], dtype=np.float32)
alphas = np.array([v[1] for v in COLOR_MAP.values()], dtype=np.float32)

# -----------------------------
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Cannot open video")

ret, frame = cap.read()
if not ret:
    raise Exception("Cannot read video")

h, w = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# -----------------------------
# LOAD MASK FILE LIST
# -----------------------------
mask_files = sorted([
    os.path.join(MASK_PATH, f)
    for f in os.listdir(MASK_PATH)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(mask_files) == 0:
    raise Exception("No mask images found in folder")

print(f"Found {len(mask_files)} masks")

# -----------------------------
# PRELOAD MASKS INTO RAM
# -----------------------------
print("Preloading masks into memory...")
masks = []
for f in mask_files:
    m = cv2.imread(f)
    if m is None:
        raise Exception(f"Cannot load mask: {f}")
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h))
    masks.append(m)

# -----------------------------
# START FFMPEG PIPE (FASTER)
# -----------------------------
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}",
    "-r", str(int(fps)),
    "-i", "-",
    "-an",
    "-vcodec", "libx264",
    "-preset", "ultrafast",   # 🔥 faster encoding
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    OUTPUT_PATH
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# -----------------------------
# PROCESS VIDEO
# -----------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx >= len(masks):
        print("Warning: fewer masks than frames")
        break

    mask = masks[frame_idx]

    # Convert once
    frame_f32 = frame.astype(np.float32)
    mask_i16 = mask.astype(np.int16)

    # -----------------------------
    # VECTORIZED COLOR MATCHING
    # -----------------------------
    diff = np.abs(mask_i16[:, :, None, :] - colors[None, None, :, :])  # (H,W,N,3)
    matches = np.all(diff < tolerance, axis=-1)                        # (H,W,N)

    idx = np.argmax(matches, axis=-1)                                  # (H,W)

    overlay_map = overlays[idx]                                        # (H,W,3)
    alpha_map = alphas[idx][..., None]                                 # (H,W,1)

    valid = np.any(matches, axis=-1)[..., None]

    result = np.where(
        valid,
        frame_f32 * (1 - alpha_map) + overlay_map * alpha_map,
        frame_f32
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    proc.stdin.write(result.tobytes())

    frame_idx += 1

# -----------------------------
# CLEANUP
# -----------------------------
proc.stdin.close()
proc.wait()
cap.release()

print("Done. Saved to:", OUTPUT_PATH)