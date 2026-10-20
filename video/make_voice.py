"""Turn video/script.md into a voiceover with Kokoro text-to-speech (open weights, runs offline on CPU).

Writes:
- video/voice.m4a, the narration;
- video/timings.json, when each scene starts. video/record.py uses it to scroll each notebook
  section for exactly as long as its narration lasts;
- video/subtitles.srt, subtitles built from Kokoro's per-word timings. video/record.py burns them into the video.

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
MAX_CUE_CHARS = 70     # longest subtitle cue, about two short lines
VIDEO_DIR = Path("video")


def read_scenes(path):
    """Return [(notebook heading, narration text)] from the '## name | heading' blocks."""
    scenes = []
    for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
        header, _, body = block.partition("\n")
        heading = header.split("|")[1].strip()
        scenes.append((heading, " ".join(body.split())))
    return scenes


def build_cues(words):
    """Group (text, start, end) words into subtitle cues.

    A cue ends at the end of a sentence, at a comma once it is half full, or when it reaches MAX_CUE_CHARS.
    """
    cues, current = [], []
    for word in words:
        current.append(word)
        text = "".join(w[0] for w in current).strip()
        last = word[0].strip()
        if last in (".", "?", "!") or (last == "," and len(text) > MAX_CUE_CHARS / 2) or len(text) >= MAX_CUE_CHARS:
            cues.append((current[0][1], current[-1][2], text))
            current = []
    if current:
        cues.append((current[0][1], current[-1][2], "".join(w[0] for w in current).strip()))
    return cues


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, start=1):
            f.write(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")


def main():
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")  # "a" = American English
    pause = np.zeros(int(PAUSE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    pieces, timings, words, t = [], [], [], 0.0

    for heading, text in read_scenes(VIDEO_DIR / "script.md"):
        timings.append({"heading": heading, "start": round(t, 2)})
        for result in pipeline(text, voice=VOICE, speed=SPEED):
            # Kokoro gives each word's start/end time within this chunk of audio; shift them to video time
            for token in result.tokens:
                if token.start_ts is not None:
                    words.append((token.text + token.whitespace, t + token.start_ts, t + token.end_ts))
            pieces.append(result.audio.numpy())
            t += len(result.audio) / SAMPLE_RATE
        pieces.append(pause)
        t += PAUSE_SECONDS
        print(f"{timings[-1]['start']:6.1f}s  {heading}")

    wav = VIDEO_DIR / "voice.wav"
    sf.write(wav, np.concatenate(pieces), SAMPLE_RATE)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(wav),
                    "-c:a", "aac", "-b:a", "128k", str(VIDEO_DIR / "voice.m4a")], check=True)
    wav.unlink()

    with open(VIDEO_DIR / "timings.json", "w") as f:
        json.dump({"scenes": timings, "end": round(t, 2)}, f, indent=2)
    cues = build_cues(words)
    write_srt(cues, VIDEO_DIR / "subtitles.srt")
    print(f"saved video/voice.m4a ({t / 60:.1f} min), video/timings.json and video/subtitles.srt ({len(cues)} cues)")


if __name__ == "__main__":
    main()
