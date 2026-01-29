import os
import cv2
import numpy as np
from pathlib import Path
from yt_dlp import YoutubeDL

def extract_frames(youtube_url, surface_tag="unsorted", base_dir="frames", max_frames=300, resize_to=(256, 256)):

    # use video ID as folder and filename label
    video_id = youtube_url.split("v=")[-1].split("&")[0]
    video_filename = f"{video_id}.mp4"

    # download video with yt-dlp
    print(f"Downloading from YouTube...")
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': video_filename,
        'quiet': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print(f"Downloaded video: {video_filename}")

    # create output directory
    output_dir = Path(base_dir) / surface_tag / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # load video and sample frames
    cap = cv2.VideoCapture(video_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = np.sort(np.random.choice(total_frames, size=min(max_frames, total_frames), replace=False))
    selected_set = set(selected_indices)

    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in selected_set:
            if resize_to:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            print(f"Saved {frame_name}")
            saved_count += 1

        frame_id += 1

    cap.release()
    os.remove(video_filename)
    print(f"Saved {saved_count} frames to '{output_dir}'")



    