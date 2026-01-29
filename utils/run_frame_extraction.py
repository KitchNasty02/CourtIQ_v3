# run_frame_extraction.py

from extract_frames import extract_frames

youtube_links = [
    ("https://www.youtube.com/watch?v=LE15sEsJZso", "hard"),
    ("https://www.youtube.com/watch?v=C4Gl-T2dtss", "hard"),
    ("https://www.youtube.com/watch?v=q_cYh3uJe_Q", "hard"),
    ("https://www.youtube.com/watch?v=I6b69yvtufI", "clay"),
    ("https://www.youtube.com/watch?v=gkIsvlZDG-Y", "clay"),
    ("https://www.youtube.com/watch?v=XOR1EuU-08A", "grass"),
    ("https://www.youtube.com/watch?v=WB5p-vx5rfE", "grass"),
]

for url, tag in youtube_links:
    extract_frames(url, surface_tag=tag)
    print(f"Extracted frames from: {url}")


    