"""Record a scrolling walkthrough of the notebook, timed to video/script.md, and add the voiceover.

Run from the project root, after the notebook has been executed:
    pip install playwright imageio-ffmpeg
    playwright install chromium
    python video/record.py

Steps:
1. export review_triage.ipynb to HTML,
2. open it in headless Chromium with video recording on, and scroll through each section on the SCENES timings,
3. merge the recording with video/voice.m4a (if it exists) into video/walkthrough.mp4.

The live Gradio demo (3:15-3:45) cannot be recorded from static HTML. Screen-record it separately and
splice it in, or narrate over the saved demo cell.
"""
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

# (start second, heading to show) - keep in sync with the timestamps in video/script.md
SCENES = [
    (0, "Review triage for a phone-accessories retailer"),
    (25, "1. Data audit"),
    (70, "2. Preprocessing"),
    (85, "3. Models"),
    (115, "4. Explainability"),
    (150, "5. Combining signals"),
    (195, "6. Live demo"),
    (225, "7. Limitations"),
]
END = 250  # total length in seconds
VIDEO_DIR = Path("video")
VOICE = VIDEO_DIR / "voice.m4a"


def export_html():
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "review_triage.ipynb",
                    "--output-dir", str(VIDEO_DIR), "--output", "notebook"], check=True)
    return (VIDEO_DIR / "notebook.html").resolve()


def heading_y(page, text):
    """Page y-coordinate of the first heading containing `text`."""
    return page.locator("h1, h2", has_text=text).first.evaluate(
        "el => el.getBoundingClientRect().top + window.scrollY")


def record(html_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 720},
                                      record_video_dir=str(VIDEO_DIR / "_raw"),
                                      record_video_size={"width": 1280, "height": 720})
        page = context.new_page()
        video_start = time.time()  # recording starts with the page; page loading is trimmed later
        page.goto(html_path.as_uri())
        page.wait_for_load_state("networkidle")

        page_bottom = page.evaluate("document.body.scrollHeight") - 720
        starts = [heading_y(page, text) - 20 for _, text in SCENES] + [page_bottom]
        times = [t for t, _ in SCENES] + [END]

        # Within each scene, scroll smoothly from its heading to just before the next one
        t0 = time.time()
        for i in range(len(SCENES)):
            y_from, y_to = starts[i], max(starts[i], starts[i + 1] - 720)
            while (elapsed := time.time() - t0) < times[i + 1]:
                frac = (elapsed - times[i]) / (times[i + 1] - times[i])
                page.evaluate(f"window.scrollTo(0, {y_from + frac * (y_to - y_from)})")
                time.sleep(0.05)

        raw_video = page.video.path()
        context.close()
        browser.close()
    return Path(raw_video), t0 - video_start


def merge(raw_video, lead_in):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = VIDEO_DIR / "walkthrough.mp4"
    cmd = [ffmpeg, "-y", "-ss", f"{lead_in:.2f}", "-i", str(raw_video)]  # skip the page-loading lead-in
    if VOICE.exists():
        cmd += ["-i", str(VOICE), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print("saved", out, "(with voiceover)" if VOICE.exists() else "(no voice.m4a found - silent)")


if __name__ == "__main__":
    merge(*record(export_html()))
