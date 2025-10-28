import os, shutil
from typing import Optional
from .config import WRITE_SUBS, COOKIES_PATH, SLEEP_REQUESTS, MAX_SLEEP_INTERVAL
from .io_utils import run, ensure_dir, basename_noext
from .captions import parse_vtt, caption_overlap, caption_keyword_density
from .features import (sample_video_histograms, detect_scenes, make_chunks, avg_motion, audio_rms, normalize, Chunk)
from .prefs import compute_weights_from_prefs, adjust_by_category

def download_video(url: str, raw_dir: str):
    ensure_dir(raw_dir)
    cmd = ["yt-dlp", "-f", "mp4", "-o", os.path.join(raw_dir, "%(id)s.%(ext)s")]
    if COOKIES_PATH:
        cmd += ["--cookies", COOKIES_PATH]
    if (SLEEP_REQUESTS and MAX_SLEEP_INTERVAL and SLEEP_REQUESTS.isdigit() and MAX_SLEEP_INTERVAL.isdigit()):
        cmd += ["--sleep-requests", SLEEP_REQUESTS, "--max-sleep-interval", MAX_SLEEP_INTERVAL]
    if WRITE_SUBS != "0":
        cmd += ["--write-auto-sub", "--sub-lang", "en", "--sub-format", "vtt"]
    cmd.append(url)
    run(cmd)
    mp4s = sorted([f for f in os.listdir(raw_dir) if f.endswith(".mp4")], key=lambda f: os.path.getsize(os.path.join(raw_dir,f)), reverse=True)
    if not mp4s: raise RuntimeError("No MP4 found after download.")
    mp4_path = os.path.join(raw_dir, mp4s[0])
    vid = basename_noext(mp4_path)
    vtt = None
    if WRITE_SUBS != "0":
        for cand in [os.path.join(raw_dir, f"{vid}.en.vtt"), os.path.join(raw_dir, f"{vid}.vtt")]:
            if os.path.exists(cand): vtt = cand; break
    return mp4_path, vtt

def compute_features_for_video(mp4_path: str, vtt_path: Optional[str], min_seg: float, max_seg: float, scene_thresh: float):
    video_id = basename_noext(mp4_path)
    captions = parse_vtt(vtt_path) if vtt_path else []
    ts, diffs = sample_video_histograms(mp4_path, fps_sample=2.0)
    if len(ts) == 0: raise RuntimeError("Failed to sample frames.")
    bounds = detect_scenes(ts, diffs, thresh=scene_thresh) or [(0.0, float(ts[-1]))]
    chunks = make_chunks(bounds, min_len=min_seg, max_len=max_seg, video_id=video_id)
    for c in chunks:
        c.motion = avg_motion(diffs, ts, c.start, c.end)
        c.audio = audio_rms(mp4_path, c.start, c.end)
        c.cap_overlap = caption_overlap(captions, c.start, c.end) if captions else 0.0
        c.kw_density = caption_keyword_density(captions, c.start, c.end) if captions else 0.0
    motions = [c.motion for c in chunks]; audios = [c.audio for c in chunks]
    texts = [0.5*c.cap_overlap + 0.5*c.kw_density for c in chunks]
    m_n = normalize(motions); a_n = normalize(audios); t_n = normalize(texts)
    for i, c in enumerate(chunks): c.score = float(0.4*m_n[i] + 0.4*a_n[i] + 0.2*t_n[i])
    return chunks

def greedy_select(chunks, target_len: float, min_gap: float):
    chosen = []; used_starts = []; total = 0.0
    for c in sorted(chunks, key=lambda x: x.score, reverse=True):
        if total >= target_len * 0.98: break
        if any(abs(c.start - s) < min_gap for s in used_starts): continue
        dur = c.end - c.start
        if total + dur > target_len + 2.0: continue
        chosen.append(c); used_starts.append(c.start); total += dur
    return chosen

def render_trailer(mp4_path: str, chunks, out_mp4: str, target_len: float, min_seg: float):
    selected = greedy_select(chunks, target_len=target_len, min_gap=min_seg/2.0)
    if not selected: raise RuntimeError("No chunks selected for trailer.")
    tmp_dir = os.path.join(os.path.dirname(out_mp4), "_tmp"); ensure_dir(tmp_dir)
    parts = []
    for i, c in enumerate(selected):
        part = os.path.join(tmp_dir, f"part_{i:03d}.mp4")
        run(["ffmpeg","-y","-ss",f"{c.start:.3f}","-to",f"{c.end:.3f}","-i", mp4_path,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k", part])
        parts.append(part)
    with open(os.path.join(tmp_dir, "files.txt"),"w",encoding="utf-8") as f:
        for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
    ensure_dir(os.path.dirname(out_mp4))
    run(["ffmpeg","-y","-safe","0","-f","concat","-i",os.path.join(tmp_dir,'files.txt'),"-c","copy", out_mp4])
    shutil.rmtree(tmp_dir, ignore_errors=True); print(f"🎬 Trailer saved: {out_mp4}")
