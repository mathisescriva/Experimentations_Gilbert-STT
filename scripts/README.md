# Scripts Directory

This directory contains utility scripts for building and managing benchmark datasets.

## build_education_v1.py

Builds the `education_v1` benchmark dataset from multiple sources.

### Usage

```bash
# Basic usage (downloads from HuggingFace)
python scripts/build_education_v1.py

# With PASTEL data from local directory
python scripts/build_education_v1.py --pastel-dir data/pastel

# Custom limits
python scripts/build_education_v1.py \
  --summre-limit 20 \
  --voxpopuli-limit 10 \
  --pastel-limit 20

# Skip validation
python scripts/build_education_v1.py --skip-validation
```

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Additional dependencies:
- `jsonlines` (for metadata.jsonl)
- `librosa` (for audio processing)
- `soundfile` (for audio I/O)

### Output

The script creates:
- `benchmark/education/audio/*.wav` - Audio files
- `benchmark/education/refs/*.txt` - Reference transcriptions
- `benchmark/education/metadata.jsonl` - Dataset metadata

### Dataset Sources

1. **SUMM-RE** (40%): Meetings and discussion-based content
   - Source: `linagora/SUMM-RE` on HuggingFace
   - Default: 15 files

2. **VoxPopuli FR** (20%): Long-form institutional speech
   - Source: `facebook/voxpopuli` (config: "fr")
   - Default: 8 files

3. **PASTEL/COCo/Canal-U** (40%): Course lectures
   - Source: Local files or manual download
   - Default: 15 files
   - See script comments for manual setup instructions

### Notes

- Audio files are normalized to 16kHz mono WAV
- Text is normalized (lowercase, no punctuation, collapsed spaces)
- Files with duration < 10s or > 10min are filtered out
- The script validates the dataset and reports statistics

