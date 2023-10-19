# BEGIN: e4f1a2d3gkq5
import re
from pytube import YouTube, Playlist
import subprocess


p = Playlist(
    "https://youtube.com/playlist?list=PLDBem4OrlfANo1KXzo4sRKqz-uflGTphY&si=1LBK2nOn7RpiCaZg"
)
for url in p.video_urls:
    yt = YouTube(url)
    print(yt.title)
    subprocess.run(["python", "scripts/video_processing.py", url])

# END: e4f1a2d3gkq5


# 19
