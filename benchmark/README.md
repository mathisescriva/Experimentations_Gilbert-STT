# Benchmark Data Structure

This directory contains the benchmark datasets for evaluating ASR models.

## Structure

Each subset (meetings, telephone, accents, longform) has two subdirectories:

- `audio/` - Audio files (.wav, .mp3, .flac, .m4a, .ogg)
- `refs/` - Reference text files (.txt)

## File Naming Convention

For each audio file, there must be a corresponding reference text file with the same base name:

```
meetings/audio/sample1.wav  →  meetings/refs/sample1.txt
meetings/audio/sample2.mp3   →  meetings/refs/sample2.txt
```

## Reference Text Format

Reference files should contain the ground truth transcription in plain text:

```
sample1.txt:
Bonjour, je suis ravi de vous rencontrer aujourd'hui. Nous allons discuter de notre projet.
```

## Adding Data

1. Place audio files in the appropriate `{subset}/audio/` directory
2. Create corresponding `.txt` files in `{subset}/refs/` with the same base name
3. Ensure text files are UTF-8 encoded

## Example

```
benchmark/
└── meetings/
    ├── audio/
    │   ├── meeting_001.wav
    │   └── meeting_002.wav
    └── refs/
        ├── meeting_001.txt  (contains: "Bonjour, bienvenue à cette réunion...")
        └── meeting_002.txt  (contains: "Nous allons commencer par...")
```

## Running the Benchmark

Once data is added, run:

```bash
python -m src.evaluation.run_benchmark --config configs/benchmark.yaml
```

See the main README.md for more details.

