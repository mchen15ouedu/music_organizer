import os, sys, re, json, shutil, subprocess, tempfile, traceback
import numpy as np

FFMPEG = r"C:\Users\chenm\anaconda3\envs\music\Library\bin\ffmpeg.exe"
SRC_ROOT = r"C:\Users\chenm\Music"
OUT_ROOT = r"C:\Users\chenm\Music\Organized"
WORK_DIR = r"C:\Users\chenm\music_organizer"
TMP_DIR  = os.path.join(WORK_DIR, "tmp")
PROGRESS = os.path.join(WORK_DIR, "progress.csv")
ERRORS   = os.path.join(WORK_DIR, "errors.log")

AUDIO_EXT = {".mp3", ".wma", ".m4a", ".flac", ".wav", ".aac", ".ogg", ".mp4"}
SR = 22050
MAX_SECONDS = 150  # analyze up to this many seconds for speed

# Krumhansl-Schmuckler key profiles
MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def decode(path):
    """Decode (a copy of) an audio file to mono float32 @ SR using ffmpeg. Never touches originals."""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-t", str(MAX_SECONDS),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or len(p.stdout) < SR:  # need at least ~1s of audio
        raise RuntimeError("ffmpeg decode failed: " + p.stderr.decode("utf-8","ignore")[:200])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()

def analyze(y):
    import librosa
    from librosa.feature.rhythm import tempo as _tempo
    # ---- tempo / BPM ----
    oenv = librosa.onset.onset_strength(y=y, sr=SR)
    bpm = float(np.atleast_1d(_tempo(onset_envelope=oenv, sr=SR, std_bpm=4))[0])
    if bpm > 0:
        while bpm < 65:  bpm *= 2
        while bpm > 185: bpm /= 2
    bpm_round = int(round(bpm)) if bpm > 0 else 0
    # ---- key (Krumhansl-Schmuckler on chroma) ----
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    prof = chroma.mean(axis=1)
    if prof.sum() > 0:
        prof = prof / prof.sum()
    best = (-2.0, None, None)  # corr, tonic, mode
    for i in range(12):
        for mode, ref in (("major", MAJ), ("minor", MIN)):
            c = np.corrcoef(np.roll(prof, -i)[::-1][::-1], None) if False else np.corrcoef(prof, np.roll(ref, i))[0,1]
            if c > best[0]:
                best = (c, NOTES[i], mode)
    corr, tonic, mode = best
    if tonic is None or corr < 0.55:
        key = "Key-Unknown"
    else:
        key = f"{tonic}-{mode}"
    return bpm_round, key, round(float(corr), 3)

# ---------- filename -> artist ----------
FEAT = re.compile(r"\s+(feat\.?|ft\.?|featuring|with)\b.*$", re.I)
def clean(s):
    s = re.sub(r"\[[^\]]*\]", "", s)          # [www...] tags
    s = re.sub(r"\([^)]*\)", "", s)           # (1), (Unplugged...) etc
    s = re.sub(r"[‐-―]", "-", s)     # unicode dashes -> hyphen
    s = re.sub(r"\.(mp3|wma|m4a|flac|wav|aac|ogg|mp4)$", "", s, flags=re.I)
    return s.strip(" -_.")

def main_artist(raw):
    name = clean(raw)
    scene = False
    m = re.match(r"^\s*\d{1,3}\s*([-_.])\s*(.*)$", name)
    if m:
        sep, rest = m.group(1), m.group(2)
        if sep == "-" and " - " not in name:
            name, scene = rest, True      # "NN-artist-title" scene release (artist first)
        else:
            name = rest                   # "NN - Title" / "NN. Title" (no artist in name)
    if " - " in name:                     # "Artist - Title"
        artist = name.split(" - ")[0]
    elif scene and "-" in name:           # scene: artist is first segment
        artist = name.split("-")[0]
    elif "-" in name:                     # "Title-Artist" (no spaces) -> artist trails
        parts = [p for p in name.split("-") if p.strip()]
        artist = parts[-1] if len(parts) >= 2 else None
    else:
        artist = None                     # title only, no artist in filename
    if not artist:
        return "Unknown-Artist"
    artist = artist.replace("_", " ")
    artist = FEAT.sub("", artist)
    artist = re.split(r"\s+(&|,|/|\bvs\b|\bx\b|\bft\b|\bfeat\b)\s+", artist, flags=re.I)[0]
    artist = re.sub(r"\s+", " ", artist).strip(" -_.")
    return artist or "Unknown-Artist"

def sanitize(s):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = s.strip(" .")
    return s[:80] if s else "Unknown"

def iter_sources():
    for root, dirs, files in os.walk(SRC_ROOT):
        if os.path.commonpath([os.path.abspath(root), os.path.abspath(OUT_ROOT)]) == os.path.abspath(OUT_ROOT):
            continue
        if os.path.abspath(root).startswith(os.path.abspath(OUT_ROOT)):
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                yield os.path.join(root, f)

def load_done():
    done = set()
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as fh:
            for line in fh:
                p = line.split("\t", 1)[0]
                if p:
                    done.add(p)
    return done

def log_err(msg):
    with open(ERRORS, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")

def process(path, done):
    if path in done:
        return "skip"
    base = os.path.basename(path)
    tmp = None
    try:
        # 1) copy original -> throwaway temp; analysis only ever reads the copy
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(base)[1], dir=TMP_DIR)
        os.close(fd)
        shutil.copy2(path, tmp)
        # 2) analyze the copy
        try:
            y = decode(tmp)
            bpm, key, conf = analyze(y)
        except Exception as e:
            bpm, key, conf = 0, "Key-Unknown", 0.0
            log_err(f"ANALYZE-FAIL\t{path}\t{e}")
        # 3) build destination from ORIGINAL filename
        bpm_folder = f"{bpm:03d}-BPM" if bpm > 0 else "BPM-Unknown"
        artist = sanitize(main_artist(base))
        dest_dir = os.path.join(OUT_ROOT, bpm_folder, artist, sanitize(key))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, base)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(base)
            n = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{stem} ({n}){ext}")
                n += 1
        # 4) copy the ORIGINAL (never decoded) into the tree
        shutil.copy2(path, dest)
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(f"{path}\t{bpm}\t{key}\t{conf}\t{artist}\t{os.path.relpath(dest, OUT_ROOT)}\n")
        return "ok"
    except Exception:
        log_err(f"FATAL\t{path}\t{traceback.format_exc()}")
        return "err"
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def main():
    global SRC_ROOT, OUT_ROOT
    args = sys.argv[1:]
    test_n = None
    limit = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--src" and i + 1 < len(args):
            SRC_ROOT = os.path.abspath(args[i + 1]); i += 2
        elif a == "--out" and i + 1 < len(args):
            OUT_ROOT = os.path.abspath(args[i + 1]); i += 2
        elif a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1]); i += 2
        elif a == "--test":
            if i + 1 < len(args) and args[i + 1].isdigit():
                test_n = int(args[i + 1]); i += 2
            else:
                test_n = 8; i += 1
        else:
            i += 1
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"SRC = {SRC_ROOT}")
    print(f"OUT = {OUT_ROOT}")
    files = list(iter_sources())
    done = load_done()
    if limit:
        files = [f for f in files if f not in done][:limit]
    if test_n:
        files = files[:test_n]
        for p in files:
            tmp = None
            try:
                fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(p)[1], dir=TMP_DIR); os.close(fd)
                shutil.copy2(p, tmp)
                y = decode(tmp); bpm, key, conf = analyze(y)
                print(f"{bpm:>4} BPM | {key:<12} (conf {conf:.2f}) | artist={main_artist(os.path.basename(p))!r} | {os.path.basename(p)}")
            except Exception as e:
                print(f"FAIL {os.path.basename(p)}: {e}")
            finally:
                if tmp and os.path.exists(tmp): os.remove(tmp)
        return
    total = len(files)
    ok = skip = err = 0
    for i, p in enumerate(files, 1):
        r = process(p, done)
        if r == "ok": ok += 1
        elif r == "skip": skip += 1
        else: err += 1
        if i % 25 == 0 or i == total:
            print(f"[{i}/{total}] ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE total={total} ok={ok} skip={skip} err={err}", flush=True)

if __name__ == "__main__":
    main()
