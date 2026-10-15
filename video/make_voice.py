"""Turn video/script.md into a voiceover with Kokoro text-to-speech (open weights, runs offline on CPU).

Writes:
- video/voice.m4a, the narration;
- video/timings.json, when each scene starts. video/record.py uses it to scroll each notebook
  section for exactly as long as its narration lasts.

Run from the project root:
    pip install kokoro soundfile imageio-ffmpeg
    python video/make_voice.py
"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from kokoro import KPipeline

VOICE = "af_heart"     # other Kokoro voices: "am_michael", "bf_emma", "bm_george"
SPEED = 1.0            # 1.1 = 10% faster
PAUSE_SECONDS = 0.8    # silence between scenes
SAMPLE_RATE = 24000    # Kokoro produces 24 kHz audio
VIDEO_DIR = Path("video")


def read_scenes(path):
    """Return [(notebook heading, narration text)] from the '## name | heading' blocks."""
    scenes = []
    for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
        header, _, body = block.partition("\n")
        heading = header.split("|")[1].strip()
        scenes.append((heading, " ".join(body.split())))
    return scenes


def main():
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")  # "a" = American English
    pause = np.zeros(int(PAUSE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    pieces, timings, t = [], [], 0.0

    for heading, text in read_scenes(VIDEO_DIR / "script.md"):
        audio = np.concatenate([r.audio.numpy() for r in pipeline(text, voice=VOICE, speed=SPEED)])
        timings.append({"heading": heading, "start": round(t, 2)})
        pieces += [audio, pause]
        t += (len(audio) + len(pause)) / SAMPLE_RATE
        print(f"{timings[-1]['start']:6.1f}s  {heading}")

    wav = VIDEO_DIR / "voice.wav"
    sf.write(wav, np.concatenate(pieces), SAMPLE_RATE)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(wav),
                    "-c:a", "aac", "-b:a", "128k", str(VIDEO_DIR / "voice.m4a")], check=True)
    wav.unlink()

    with open(VIDEO_DIR / "timings.json", "w") as f:
        json.dump({"scenes": timings, "end": round(t, 2)}, f, indent=2)
    print(f"saved video/voice.m4a ({t / 60:.1f} min) and video/timings.json")


if __name__ == "__main__":
    main()
