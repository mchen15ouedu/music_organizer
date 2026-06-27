"""
Music organizer v2 — online/identification-first.

For each file:
  1. Identify the song from its FILE NAME using OpenAI (great at parsing messy names
     and recalling well-known songs' BPM/key).
  2. If the model confidently recognizes it -> use its BPM / key / artist.
  3. If the name is garbled / non-text, or the model doesn't know the song ->
     fall back to the audio-analysis script (librosa) from organize.py.

Organizes (copy, never move) into  Music/Organized_v2/<BPM>/<Artist>/<Key>/.

Run with the 'music' conda env python (has librosa + ffmpeg + requests).
    python organize_v2.py --sample 40     # dry-run: print decisions for a spread of files
    python organize_v2.py                  # full run (copies files)
    python organize_v2.py --limit 100      # process first 100 only
"""
import os
import sys
import re
import json
import shutil
import tempfile
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import organize as O   # reuse decode / analyze / _safe / sanitize / main_artist

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SRC_ROOT = r"C:\Users\chenm\Music"
OUT_ROOT = r"C:\Users\chenm\Music\Organized_v2"
PROGRESS = r"C:\Users\chenm\music_organizer\progress_v2.csv"
BATCH = 25


def iter_sources():
    """All audio files under Music, excluding any Organized* output tree."""
    for root, dirs, files in os.walk(SRC_ROOT):
        dirs[:] = [d for d in dirs if not d.lower().startswith("organized")]
        for f in files:
            if os.path.splitext(f)[1].lower() in O.AUDIO_EXT:
                yield os.path.join(root, f)


def identify_batch(filenames):
    """OpenAI: identify songs + bpm/key from a batch of file names. Returns list of dicts."""
    listing = "\n".join(f"{i}. {fn}" for i, fn in enumerate(filenames))
    prompt = (
        "You are given music FILE NAMES (often messy: leading track numbers, website tags, "
        "'feat.'/'ft.', different languages, underscores). For each item identify the song.\n"
        "Return ONLY a JSON array, one object per input item, fields:\n"
        '  "i": the index number,\n'
        '  "artist": main artist (string) or null,\n'
        '  "title": song title (string) or null,\n'
        '  "recognizable": true ONLY if this is a real song you can identify from the name,\n'
        '  "bpm": integer tempo of the song if you know it (you know most well-known songs), else null,\n'
        '  "key": musical key like "C# minor" if you know it, else null,\n'
        '  "confidence": "high" | "medium" | "low".\n'
        "Rules: numeric/garbled/non-song names -> recognizable=false, bpm=null, key=null. "
        "For songs you recognize, DO fill in bpm and key from your knowledge; use null only when "
        "you genuinely don't know the specific song.\n\n"
        + listing
    )
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                      json={"model": OPENAI_MODEL, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]},
                      timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    s, e = txt.find("["), txt.rfind("]")
    return json.loads(txt[s:e + 1])


def normalize_key(k):
    if not k:
        return "Key-Unknown"
    k = str(k).strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g][#b]?)\s*[-\s]?\s*(maj|min|major|minor)", k, re.I)
    if not m:
        return "Key-Unknown"
    note = m.group(1)[0].upper() + m.group(1)[1:]
    mode = "minor" if m.group(2).lower().startswith("min") else "major"
    return f"{note}-{mode}"


def fallback_analyze(path):
    """librosa fallback -> (bpm, key)."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(path)[1], dir=O.TMP_DIR)
        os.close(fd)
        shutil.copy2(path, tmp)
        y = O.decode(tmp)
        bpm, key, _ = O.analyze(y)
        return bpm, key
    except Exception:
        return 0, "Key-Unknown"
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def decide(path, info):
    """Return (bpm, key, artist, source) using LLM info first, else analysis."""
    base = os.path.basename(path)
    artist = (info.get("artist") or "").strip() or O.main_artist(base)
    bpm_llm = info.get("bpm")
    # The model returns a bpm only for songs it actually knows; that's our signal to trust it.
    if info.get("recognizable") and isinstance(bpm_llm, (int, float)) and bpm_llm > 0:
        return int(round(bpm_llm)), normalize_key(info.get("key")), artist, "online"
    bpm, key = fallback_analyze(path)
    return bpm, key, artist, "analysis"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_done():
    done = set()
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as fh:
            for line in fh:
                p = line.split("\t", 1)[0]
                if p:
                    done.add(p)
    return done


def file_into_tree(path, bpm, key, artist):
    bpm_folder = f"{bpm:03d}-BPM" if bpm and bpm > 0 else "BPM-Unknown"
    dest_dir = os.path.join(OUT_ROOT, bpm_folder, O.sanitize(artist), O.sanitize(key))
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(path)
    dest = os.path.join(dest_dir, base)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(base)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{stem} ({n}){ext}")
            n += 1
    shutil.copy2(path, dest)
    return os.path.relpath(dest, OUT_ROOT)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    sample = limit = None
    if "--sample" in args:
        sample = int(args[args.index("--sample") + 1])
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if not OPENAI_KEY:
        sys.exit("OPENAI_API_KEY not set.")
    os.makedirs(O.TMP_DIR, exist_ok=True)

    files = sorted(iter_sources())
    if sample:                       # spread across the whole library for a representative preview
        step = max(1, len(files) // sample)
        files = files[::step][:sample]
        print(f"DRY-RUN preview of {len(files)} files spread across the library:\n")
        print(f"{'BPM':>4} {'src':<9} {'key':<10} {'artist':<22} file")
        counts = {"online": 0, "analysis": 0}
        for batch in chunks(files, BATCH):
            names = [os.path.basename(p) for p in batch]
            try:
                ids = identify_batch(names)
            except Exception as ex:
                print("  (LLM batch failed:", ex, ")")
                ids = []
            by_i = {d.get("i"): d for d in ids}
            for j, p in enumerate(batch):
                bpm, key, artist, src = decide(p, by_i.get(j, {}))
                counts[src] = counts.get(src, 0) + 1
                print(f"{bpm:>4} {src:<9} {key:<10} {artist[:21]:<22} {os.path.basename(p)[:50]}")
        print(f"\nSource split: online (OpenAI) = {counts['online']}, "
              f"audio-analysis fallback = {counts['analysis']}")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    done = load_done()
    files = [f for f in files if f not in done]
    if limit:
        files = files[:limit]
    total = len(files)
    ok = err = 0
    for bi, batch in enumerate(chunks(files, BATCH)):
        names = [os.path.basename(p) for p in batch]
        try:
            ids = identify_batch(names)
        except Exception:
            ids = []
        by_i = {d.get("i"): d for d in ids}
        for j, p in enumerate(batch):
            try:
                bpm, key, artist, src = decide(p, by_i.get(j, {}))
                rel = file_into_tree(p, bpm, key, artist)
                with open(PROGRESS, "a", encoding="utf-8") as fh:
                    fh.write(f"{p}\t{bpm}\t{key}\t{artist}\t{src}\t{rel}\n")
                ok += 1
            except Exception as ex:
                err += 1
                with open(os.path.join(O.WORK_DIR, "errors_v2.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"{p}\t{ex}\n")
        print(f"[{ok + err}/{total}] ok={ok} err={err}", flush=True)
    print(f"DONE total={total} ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
