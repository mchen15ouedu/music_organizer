# Music Organizer

Organizes a music library into a `BPM / Artist / Key` folder tree by analyzing the
audio directly — no reliance on (often missing/unreliable) online BPM databases.

Files are **copied**, never moved or deleted, so the originals are always preserved.
Analysis is run only on a throwaway temp copy of each file, so the originals are
never opened by the decoder either.

## Output layout

```
Organized/
  096-BPM/
    Ed Sheeran/
      C#-minor/
        song.mp3
```

- **BPM** — detected from the waveform (librosa, tuned tempo prior), rounded to an integer.
- **Artist** — parsed from the filename (handles `Artist - Title`, `Title-Artist`,
  and underscore "scene release" names where the artist comes first).
- **Key** — detected via chroma analysis (Krumhansl-Schmuckler). Low-confidence
  results go to `Key-Unknown` rather than being guessed; the song is still filed by BPM + artist.
- Files that can't be decoded land in `BPM-Unknown` so nothing is lost.

## Requirements

A conda environment with librosa + ffmpeg:

```
conda create -n music -c conda-forge python=3.12 librosa ffmpeg -y
```

> The `FFMPEG` path near the top of `organize.py` points at the conda env's
> `ffmpeg.exe`. Adjust it if your env lives elsewhere.

## Usage

```
conda activate music
python organize.py                       # organize the default source folder
python organize.py --src "D:\New Songs"  # organize a specific folder
python organize.py --out "E:\Sorted"     # write the tree somewhere else
python organize.py --test 5              # dry run: print BPM/key/artist for 5 files
```

On Windows you can also double-click `Organize-Music.bat`, or drag a folder onto it
to organize that folder. Re-running only processes files not already recorded in
`progress.csv`, so adding new music is incremental.

## Notes

- Automated tempo detection is ~85–90% accurate; key detection is the softer axis
  (~60–70% for the confident ones). BPM is the reliable sort key.
- Paths in `organize.py` / `Organize-Music.bat` are absolute and personal — edit
  them for your own machine.
