import pandas as pd
import av
import sys

df = pd.read_csv("/kwkj-k8s/LTX-2/videos-lty/0331seedance/seedance原视频_25fps_121frames.csv")
for i, row in df.head(5).iterrows():
    path = row["media_path"]
    try:
        container = av.open(path)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        print(f"{path}: {len(audio_streams)} audio streams")
    except Exception as e:
        print(f"{path}: error {e}")
