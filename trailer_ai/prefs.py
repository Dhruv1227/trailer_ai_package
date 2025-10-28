from typing import Dict, Tuple
def compute_weights_from_prefs(prefs: Dict[str, str]) -> Tuple[float,float,float]:
    style = prefs.get("style","action"); focus = prefs.get("focus","both")
    w_motion, w_audio, w_text = 0.4, 0.4, 0.2
    if style == "action": w_motion, w_audio, w_text = 0.55, 0.35, 0.10
    elif style == "emotional": w_motion, w_audio, w_text = 0.20, 0.30, 0.50
    elif style == "funny": w_motion, w_audio, w_text = 0.30, 0.25, 0.45
    elif style == "informative": w_motion, w_audio, w_text = 0.10, 0.20, 0.70
    if focus == "dialogue": w_text += 0.15
    elif focus == "visuals": w_motion += 0.15
    total = w_motion + w_audio + w_text
    return (w_motion/total, w_audio/total, w_text/total)
def _normalize_weights(w_motion, w_audio, w_text):
    total = w_motion + w_audio + w_text
    if total <= 1e-9:
        return 0.34, 0.33, 0.33
    return (w_motion/total, w_audio/total, w_text/total)
def adjust_by_category(category: str, w_motion: float, w_audio: float, w_text: float, base_min: float, base_max: float):
    cat = (category or "").strip().lower()
    min_seg, max_seg = base_min, base_max
    if cat in ("sports",): w_motion += 0.20; w_audio += 0.10; w_text -= 0.10; min_seg = max(1.5, base_min*0.85); max_seg = max(3.0, base_max*0.85)
    elif cat in ("nature",): w_motion -= 0.10; w_audio -= 0.10; w_text += 0.20; min_seg = base_min*1.25; max_seg = base_max*1.35
    elif cat in ("gaming",): w_motion += 0.15; w_audio += 0.05; w_text -= 0.10; min_seg = max(1.5, base_min*0.8); max_seg = max(3.5, base_max*0.8)
    elif cat in ("music",): w_audio += 0.35; w_motion -= 0.10
    elif cat in ("education", "howto", "how-to", "tutorial"): w_text += 0.35; w_motion -= 0.10; min_seg = base_min*1.15; max_seg = base_max*1.15
    elif cat in ("news", "interview"): w_text += 0.25; w_audio += 0.05
    elif cat in ("travel", "vlog"): w_motion += 0.05; w_text += 0.05; min_seg = base_min*1.05; max_seg = base_max*1.15
    elif cat in ("cooking",): w_text += 0.20; w_audio -= 0.05; min_seg = base_min*1.10; max_seg = base_max*1.10
    elif cat in ("finance", "tech"): w_text += 0.25; w_motion -= 0.05; min_seg = base_min*1.10; max_seg = base_max*1.10
    w_motion, w_audio, w_text = _normalize_weights(max(0.0,w_motion), max(0.0,w_audio), max(0.0,w_text))
    return w_motion, w_audio, w_text, float(min_seg), float(max_seg)
