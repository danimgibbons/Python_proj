import os
import random
import subprocess
import json
import csv

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "/Users/danimgibbons/Desktop/PhD/skater_proj/vids/chunks_Dani/"
FPS = 50
CUT_FRAMES = 500
SEED = 42

FILES = [
    os.path.join(BASE_DIR, "video1/big1_small2.mp4"),
    os.path.join(BASE_DIR, "video2/cleaned/big1_small2_cleaned.mp4"),
    os.path.join(BASE_DIR, "video3/cleaned/big1_small2_cleaned.mp4"),
]

random.seed(SEED)

# -----------------------------
# Get duration
# -----------------------------
def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd)
    return float(json.loads(out)["streams"][0]["duration"])

duration = get_duration(FILES[0])
total_frames = int(duration * FPS)

# -----------------------------
# Generate cut plan
# -----------------------------
cams = [0, 1, 2]
current_cam = random.choice(cams)

segments = []
frame = 0

while frame < total_frames:
    end = min(frame + CUT_FRAMES, total_frames)
    segments.append((frame, end, current_cam))
    current_cam = random.choice([c for c in cams if c != current_cam])
    frame = end

# -----------------------------
# Build camera log (FRAME-LEVEL)
# -----------------------------
camera_log = [0] * total_frames

for start, end, cam in segments:
    for f in range(start, end):
        camera_log[f] = cam

# -----------------------------
# Write camera log for MATLAB
# -----------------------------
output_video = os.path.join(BASE_DIR, f"output_fixed_{CUT_FRAMES}f.mp4")
output_log = os.path.join(BASE_DIR, f"output_fixed_{CUT_FRAMES}f_camlog.csv")

with open(output_log, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "camera"])
    for i, cam in enumerate(camera_log):
        writer.writerow([i + 1, cam])

# -----------------------------
# Build filter_complex
# -----------------------------
filters = []
concat_inputs = []

for i, (start, end, cam) in enumerate(segments):
    t_start = start / FPS
    t_end = end / FPS
    label = f"v{i}"

    filters.append(
        f"[{cam}:v]"
        f"trim=start={t_start}:end={t_end},"
        f"setpts=PTS-STARTPTS"
        f"[{label}]"
    )
    concat_inputs.append(f"[{label}]")

filter_complex = (
    ";\n".join(filters)
    + ";\n"
    + "".join(concat_inputs)
    + f"concat=n={len(segments)}:v=1:a=0[outv]"
)

# -----------------------------
# Run FFmpeg
# -----------------------------
cmd = [
    "ffmpeg", "-y",
    "-i", FILES[0],
    "-i", FILES[1],
    "-i", FILES[2],
    "-filter_complex", filter_complex,
    "-map", "[outv]",
    "-r", str(FPS),
    output_video
]

subprocess.run(cmd, check=True)

print("Saved video:", output_video)
print("Saved cam log:", output_log)