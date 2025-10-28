import os, re
from typing import List, Tuple
def seconds_from_vtt_ts(ts: str) -> float:
    parts = re.split(r"[,:.]", ts)
    if len(parts) < 3:
        return 0.0
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s
def parse_vtt(vtt_path: str):
    if not vtt_path or not os.path.exists(vtt_path):
        return []
    entries = []
    with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if block:
                    for i, ln in enumerate(block):
                        if "-->" in ln:
                            t1, t2 = [x.strip() for x in ln.split("-->")]
                            try:
                                s = seconds_from_vtt_ts(t1)
                                e = seconds_from_vtt_ts(t2.split(" ")[0])
                                text = " ".join(block[i+1:]).strip()
                                if e > s:
                                    entries.append((s, e, text))
                            except Exception:
                                pass
                            break
                block = []
            else:
                block.append(line)
        if block:
            for i, ln in enumerate(block):
                if "-->" in ln:
                    t1, t2 = [x.strip() for x in ln.split("-->")]
                    try:
                        s = seconds_from_vtt_ts(t1)
                        e = seconds_from_vtt_ts(t2.split(" ")[0])
                        text = " ".join(block[i+1:]).strip()
                        if e > s:
                            entries.append((s, e, text))
                    except Exception:
                        pass
                    break
    return entries
def caption_overlap(captions, start: float, end: float) -> float:
    if not captions:
        return 0.0
    dur = max(1e-6, end - start)
    covered = 0.0
    for s, e, _ in captions:
        inter = max(0.0, min(end, e) - max(start, s))
        covered += inter
    covered = min(covered, dur)
    return covered / dur
def caption_keyword_density(captions, start: float, end: float) -> float:
    import re
    tot_words = 0.0
    dur = max(1e-6, end - start)
    for s, e, text in captions:
        inter = max(0.0, min(end, e) - max(start, s))
        if inter > 0:
            w = len(re.findall(r"\w+", text.lower()))
            tot_words += w * (inter / (e - s + 1e-6))
    return float(tot_words / dur)
