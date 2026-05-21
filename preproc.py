import subprocess
import numpy as np
from scipy.optimize import minimize_scalar
import os

# ---------------------------
# CONFIGURATION
# ---------------------------

video_files = [
    "/Users/danimgibbons/Desktop/Skater/05nov_CAM_01_V2.mp4",
    "/Users/danimgibbons/Desktop/Skater/05nov_CAM_02_V2.mp4",
    "/Users/danimgibbons/Desktop/Skater/05nov_CAM_03_V2.mp4"
]

first_frames = [27777, 27775, 27776]
fps = 50

start_timecodes = [
    "00:09:15", "00:19:54","00:28:05", "00:35:20",
    "00:43:22", "00:54:00","01:01:21", "01:08:50",
    "01:17:11", "01:25:43","01:35:01","01:44:26","01:52:34"
]

end_timecodes = [
    "00:17:49","00:26:52","00:34:04","00:40:22","00:49:32",
    "01:00:16","01:07:32","01:15:17","01:23:30","01:32:06",
    "01:41:20","01:50:46","01:58:49"
]

output_path = "/Users/danimgibbons/Desktop/Skater/chunks_Dani/"
os.makedirs(output_path, exist_ok=True)

# ---------------------------
# FUNCTIONS
# ---------------------------

def timecode_to_seconds(tc, fps):
    parts = list(map(int, tc.split(":")))
    h, m, s = parts[:3]
    f = parts[3] if len(parts) == 4 else 0
    return h * 3600 + m * 60 + s + f / fps

def cut_big_chunk(video_file, start_sec, end_sec, output_file):
    command = [
        "ffmpeg",
        "-ss", str(start_sec),       # FAST SEEK
        "-to", str(end_sec),
        "-i", video_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_file
    ]
    subprocess.run(command, check=True)

def cut_small_chunks(video_file, duration, small_chunk_duration, output_pattern):
    num_chunks = int(np.floor(duration / small_chunk_duration))
    for j in range(num_chunks):
        cs = j * small_chunk_duration
        ce = cs + small_chunk_duration
        output_file = output_pattern.format(j=j+1)
        command = [
            "ffmpeg",
            "-ss", str(cs),
            "-to", str(ce),
            "-i", video_file,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            output_file
        ]
        subprocess.run(command, check=True)

# ---------------------------
# MAIN SCRIPT
# ---------------------------

# Base absolute times
base_start_seconds = np.array(
    [timecode_to_seconds(tc, fps) for tc in start_timecodes]
) + 4

base_end_seconds = np.array(
    [timecode_to_seconds(tc, fps) for tc in end_timecodes]
)

ref_first_frame = first_frames[0]

start_seconds_all = []
end_seconds_all = []

for first_frame in first_frames:
    offset = (first_frame - ref_first_frame) / fps
    start_seconds_all.append((base_start_seconds + offset).tolist())
    end_seconds_all.append((base_end_seconds + offset).tolist())

# Optimal small chunk duration (unchanged)
big_chunk_durations = (
    np.array(end_seconds_all[0]) - np.array(start_seconds_all[0])
)

fn = lambda x: sum(big_chunk_durations % x)
bestx_result = minimize_scalar(fn, bounds=[120, 180], method="bounded")
small_chunk_duration = bestx_result.x

print(f"Optimal small chunk duration: {small_chunk_duration:.2f}s")

# ---------------------------
# PROCESS VIDEOS
# ---------------------------

for vid_idx, video_file in enumerate(video_files):
    video_folder = os.path.join(output_path, f"video{vid_idx+1}")
    os.makedirs(video_folder, exist_ok=True)

    for chunk_idx, (s, e) in enumerate(
        zip(start_seconds_all[vid_idx], end_seconds_all[vid_idx])
    ):
        big_chunk_file = os.path.join(
            video_folder, f"big{chunk_idx+1}.mp4"
        )

        # 1 Cut big chunk ONCE
        cut_big_chunk(video_file, s, e, big_chunk_file)

        # 2 Cut small chunks FROM big chunk
        cut_small_chunks(
            video_file=big_chunk_file,
            duration=(e - s),
            small_chunk_duration=small_chunk_duration,
            output_pattern=os.path.join(
                video_folder, f"big{chunk_idx+1}_small{{j}}.mp4"
            )
        )
