from dataclasses import dataclass
from typing import List
import numpy as np, cv2, librosa
def normalize(vals):
    import numpy as np
    a = np.asarray(vals, dtype=np.float32)
    if a.size == 0:
        return a
    mn, mx = float(np.min(a)), float(np.max(a))
    if mx - mn < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)
def sample_video_histograms(mp4_path: str, fps_sample: float = 2.0):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / fps_sample)))
    ts_list, diffs = [], []
    prev_hist = None
    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h0 = cv2.calcHist([hsv], [0], None, [64], [0, 180])
            h1 = cv2.calcHist([hsv], [1], None, [64], [0, 256])
            h2 = cv2.calcHist([hsv], [2], None, [64], [0, 256])
            hist = np.concatenate([h0.flatten(), h1.flatten(), h2.flatten()]).astype(np.float32)
            hist /= (np.sum(hist) + 1e-6)
            if prev_hist is None:
                diffs.append(0.0)
            else:
                diffs.append(float(np.sum(np.abs(hist - prev_hist))))
            prev_hist = hist
            ts_list.append(ts)
        frame_idx += 1
    cap.release()
    ts = np.array(ts_list, dtype=np.float32)
    diffs = np.array(diffs, dtype=np.float32)
    return ts, diffs
def detect_scenes(ts, diffs, thresh: float = 0.55):
    if len(ts) == 0:
        return [(0.0, 0.0)]
    cuts = [0]
    for i in range(1, len(diffs)):
        if diffs[i] > thresh:
            cuts.append(i)
    cuts.append(len(ts) - 1)
    bounds = []
    for i in range(len(cuts) - 1):
        s = float(ts[cuts[i]]); e = float(ts[cuts[i+1]])
        if e > s:
            bounds.append((s, e))
    return bounds
@dataclass
class Chunk:
    video_id: str
    start: float
    end: float
    motion: float = 0.0
    audio: float = 0.0
    cap_overlap: float = 0.0
    kw_density: float = 0.0
    score: float = 0.0
def make_chunks(bounds: List[tuple], min_len=2.0, max_len=6.0, video_id="vid"):
    chunks: List[Chunk] = []
    for s, e in bounds:
        cur = s
        while cur + min_len <= e:
            end = min(cur + max_len, e)
            chunks.append(Chunk(video_id=video_id, start=cur, end=end))
            cur += min_len
    return chunks
def avg_motion(diffs, ts, start, end) -> float:
    import numpy as np
    mask = (ts >= start) & (ts <= end)
    if not np.any(mask):
        return 0.0
    return float(np.mean(diffs[mask]))
def audio_rms(mp4_path: str, start: float, end: float) -> float:
    dur = max(0.0, end - start)
    if dur <= 0.0:
        return 0.0
    try:
        y, sr = librosa.load(mp4_path, sr=None, offset=max(0.0, start), duration=dur)
        if y.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(y**2)))
    except Exception:
        return 0.0
