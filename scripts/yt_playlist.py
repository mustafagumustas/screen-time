# BEGIN: e4f1a2d3gkq5
import re
from pytube import YouTube, Playlist
import subprocess


playlist_url = input("Enter the YouTube playlist URL: ")
p = Playlist(playlist_url)

for url in p.video_urls:
    yt = YouTube(url)
    print(yt.title)
    if int((yt.title.split())[5].split(".")[0]) > 2:
        print(yt.title)
        print(url)
        subprocess.run(
            [
                "python",
                "/Users/mustafagumustas/screen-time/scripts/video_processing.py",
                url,
            ]
        )

# END: e4f1a2d3gkq5


# 19
