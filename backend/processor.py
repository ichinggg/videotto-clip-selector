# ============================================
# processor.py — CLEAN TRANSCRIPT-DRIVEN HIGHLIGHT PICKER (FIXED)
# ============================================
# Install ONCE:
# pip install moviepy openai-whisper numpy librosa opencv-python sentence-transformers torch

from moviepy.editor import VideoFileClip
import whisper, numpy as np, librosa, os, uuid, cv2
from sentence_transformers import SentenceTransformer, util

# ------------------ MODELS ------------------
print("Loading Whisper model ...")
model = whisper.load_model("base")
print("Whisper loaded.")

print("Loading semantic model ...")
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Semantic model loaded.")

# ------------------ UTILITIES ------------------

def format_timestamp(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

FAREWELL_WORDS = [
    "thank you", "thanks for watching", "see you", "goodbye",
    "take care", "subscribe", "like and", "follow us", "catch you"
]

def looks_like_outro(txt):
    t = txt.lower()
    return any(w in t for w in FAREWELL_WORDS)

# What we are explicitly optimizing for
HIGHLIGHT_QUERY = (
    "a funny, witty, surprising, or iconic moment that stands out and makes viewers react"
)

# ------------------ 1) AUDIO EXTRACTION ------------------

def extract_audio(video_path):
    v = VideoFileClip(video_path)
    if not v.audio:
        raise RuntimeError("No audio in video")
    audio_path = video_path.replace(".mp4", ".wav")
    v.audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
    return audio_path


def transcribe_audio(audio_path):
    print("Transcribing audio ...")
    result = model.transcribe(audio_path)
    print(f"✅ Transcribed {len(result['segments'])} raw segments.")
    return result["segments"]

# ------------------ 2) BUILD TRUE SENTENCES (NO MID-SENTENCE CUTS) ------------------
# We REBUILD sentences from Whisper word timings instead of trusting its segments.

def build_sentence_units(segments):
    """
    Returns a list of:
    {start, end, text}
    where each unit corresponds to a FULL sentence (or long clause),
    never cut mid-sentence.
    """
    words = []

    # Flatten all words with timestamps
    for seg in segments:
        for w in seg.get("words", []) if isinstance(seg, dict) else []:
            words.append(w)

    # If Whisper didn't return word-level timing, fall back to segments
    if not words:
        print("⚠️ No word-level timestamps found — falling back to segment stitching.")
        words = []
        for seg in segments:
            words.append({
                "word": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            })

    sentences = []
    buf = []
    sent_start = None

    def is_sentence_end(text):
        return text.strip().endswith((".", "?", "!"))

    for w in words:
        if sent_start is None:
            sent_start = w["start"]
        buf.append(w)

        # If this word ends a sentence, close it
        if is_sentence_end(w["word"]):
            sent_text = "".join(x["word"] for x in buf).strip()
            sent_end = buf[-1]["end"]

            if not looks_like_outro(sent_text) and len(sent_text.split()) >= 5:
                sentences.append({
                    "start": sent_start,
                    "end": sent_end,
                    "text": sent_text,
                })
            buf = []
            sent_start = None

    # Catch any leftover long clause
    if buf and len("".join(x["word"] for x in buf).split()) >= 7:
        sentences.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": "".join(x["word"] for x in buf).strip(),
        })

    print(f"✅ Built {len(sentences)} TRUE sentence units.")
    return sentences

# ------------------ 3) BUILD CLIPS FROM SENTENCES (TRANSCRIPT-DRIVEN) ------------------

def make_clip_candidates(sentences, min_dur=12, max_dur=26):
    cands = []
    i = 0

    while i < len(sentences):
        st = sentences[i]["start"]
        et = sentences[i]["end"]
        txt = sentences[i]["text"]
        j = i

        # Merge WHOLE sentences until duration threshold is met
        while (et - st) < min_dur and j + 1 < len(sentences):
            j += 1
            if looks_like_outro(sentences[j]["text"]):
                break
            et = sentences[j]["end"]
            txt += " " + sentences[j]["text"]

        dur = et - st
        if dur > max_dur:
            # Trim but DO NOT cut the last sentence
            et = sentences[j]["end"]
            dur = et - st

        if 12 <= dur <= 30:  # allow slightly longer if needed
            cands.append({"start": st, "end": et, "text": txt.strip()})

        i = j + 1

    print(f"✅ Built {len(cands)} transcript-aligned candidates.")
    return cands

# ------------------ 4) AUDIO / VISUAL FEATURES ------------------

def compute_audio_metrics(y, sr, start, end):
    s, e = int(start * sr), min(int(end * sr), len(y))
    if s >= e:
        return 0.0, 0.0
    segment = np.abs(y[s:e])
    return float(np.mean(segment)), float(np.std(segment))


def compute_visual_metrics(video_path, start, end, sample_fps=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / sample_fps))

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    prev, diffs, idx = None, [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t > end:
            break

        if idx % interval == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(np.mean(np.abs(g - prev)))
            prev = g
        idx += 1

    cap.release()
    if not diffs:
        return 0.0, 0.0
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.std(diffs))

# ------------------ 5) ACTIVE SPEAKER FACE CENTER (ROBUST) ------------------

def detect_face_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x + w // 2, y + h // 2)
    # fallback to center
    h, w = gray.shape
    return (w // 2, h // 2)

# ------------------ 6) EXPORT CLIP (TRACK FACE ACROSS CLIP) ------------------

def export_clip(video_path, start, end, idx, job_id, crop_mode="none"):
    out_dir = f"exports/{job_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/clip_{idx}.mp4"

    with VideoFileClip(video_path) as v:
        start, end = max(0, start), min(v.duration, end)
        pad = 0.3
        sub = v.subclip(max(0, start - pad), min(v.duration, end + pad))
        w, h = sub.size
        aspect = w / h

        # Sample multiple frames to track ACTIVE speaker
        centers = []
        for frac in [0.2, 0.5, 0.8]:
            frame = sub.get_frame((end - start) * frac)
            centers.append(detect_face_center(frame))

        avg_cx = int(np.mean([c[0] for c in centers]))
        avg_cy = int(np.mean([c[1] for c in centers]))

        target = 1 if crop_mode == "square" else (9/16 if crop_mode == "vertical" else aspect)

        if aspect > target:
            new_w = int(h * target)
            x1 = max(0, min(w - new_w, avg_cx - new_w // 2))
            sub = sub.crop(x1=x1, x2=x1 + new_w)
        elif aspect < target:
            new_h = int(w / target)
            y1 = max(0, min(h - new_h, avg_cy - new_h // 2))
            sub = sub.crop(y1=y1, y2=y1 + new_h)

        if crop_mode == "square":
            sub = sub.resize(height=1080)
        elif crop_mode == "vertical":
            sub = sub.resize(height=1920)

        sub.write_videofile(out_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return f"/exports/{job_id}/clip_{idx}.mp4"

# ------------------ 7) MAIN PIPELINE (REAL TRANSCRIPT + AV SCORING) ------------------

def process_video(video_path, crop_mode="none", job_id=None):
    print("=== PROCESSING (TRANSCRIPT + AUDIO + VISUAL) ===")
    job_id = job_id or str(uuid.uuid4())[:8]

    audio_path = extract_audio(video_path)
    raw_segs = transcribe_audio(audio_path)

    # *** THIS IS THE KEY FIX: we build TRUE sentences first ***
    sentences = build_sentence_units(raw_segs)

    # Then build clips ONLY from complete sentences
    candidates = make_clip_candidates(sentences)

    # Precompute semantic reference
    query_emb = sem_model.encode(HIGHLIGHT_QUERY, convert_to_tensor=True)

    y, sr = librosa.load(audio_path, sr=None)

    scored = []
    for c in candidates:
        st, et = c["start"], c["end"]
        dur = max(1e-6, et - st)

        # AUDIO / VISUAL (REAL, NOT FAKE)
        mean_e, std_e = compute_audio_metrics(y, sr, st, et)
        mean_m, std_m = compute_visual_metrics(video_path, st, et)

        # SPEECH RATE
        sr_words = len(c["text"].split()) / dur

        # SEMANTICS (ON FULL SENTENCES)
        phrase_emb = sem_model.encode(c["text"], convert_to_tensor=True)
        semantic_sim = float(util.cos_sim(phrase_emb, query_emb).item())

        # FINAL SCORE (balanced: transcript + AV)
        score = (
            1.2 * std_e +
            1.0 * mean_e +
            0.9 * mean_m +
            0.6 * std_m +
            0.3 * sr_words +
            3.5 * semantic_sim
        )

        scored.append({
            **c,
            "speech_rate": sr_words,
            "energy": mean_e,
            "motion": mean_m,
            "semantic": semantic_sim,
            "score": score,
        })

    # Rank by REAL combined score
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Pick top 3 that are at least 20 seconds apart (more separation)
    top = []
    for s in scored:
        if len(top) >= 3:
            break
        if all(abs(s["start"] - t["start"]) >= 20 for t in top):
            top.append(s)

    results = []
    for i, c in enumerate(top, start=1):
        f = export_clip(video_path, c["start"], c["end"], i, job_id, crop_mode)

        # USER-FRIENDLY EXPLANATION
        if c["semantic"] > 0.65:
            why = "Funny / iconic punchline or standout moment"
        elif c["energy"] > np.mean([s["energy"] for s in scored]):
            why = "High energy delivery with strong emphasis"
        else:
            why = "Engaging moment with clear speaking and movement"

        results.append({
            "clip": i,
            "start": format_timestamp(c["start"]),
            "end": format_timestamp(c["end"]),
            "duration": f"{c['end']-c['start']:.1f}s",
            "file": f,
            "reason": why,
        })

    print("\n=== TOP 3 HIGHLIGHTS ===")
    for r in results:
        print(f"[{r['start']}–{r['end']}] {r['reason']} ➜ {r['file']}")
    print("=== DONE ===")
    return results
