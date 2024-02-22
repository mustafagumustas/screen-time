import re
import sys
from pytube import YouTube, Playlist
import subprocess

# Check if a URL is provided as a command-line argument
if len(sys.argv) > 1:
    playlist_url = sys.argv[1]
else:
    print("No playlist URL provided.")
    sys.exit(1)

p = Playlist(playlist_url)

for url in p.video_urls:
    yt = YouTube(url)
    print(yt.title)
    subprocess.run(
        [
            "python",
            "screen-time/scripts/video_processing.py",
            url,
        ]
    )
