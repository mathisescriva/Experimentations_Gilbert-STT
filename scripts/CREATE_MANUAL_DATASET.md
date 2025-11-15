# 📝 Créer le Dataset education_v1 Manuellement

Si vous n'avez pas assez d'espace disque pour télécharger automatiquement, voici comment créer le dataset manuellement :

## Option 1 : Utiliser des Fichiers Audio Existants

Si vous avez déjà des fichiers audio éducatifs :

1. **Placer les fichiers audio** :
   ```bash
   # Copier vos fichiers audio
   cp /path/to/your/audio/*.wav benchmark/education/audio/
   ```

2. **Créer les transcriptions** :
   ```bash
   # Option A : Utiliser un modèle pour générer des pseudo-références
   python -m src.evaluation.prepare_references model \
     --audio-dir benchmark/education/audio \
     --output-refs-dir benchmark/education/refs \
     --model-name "openai/whisper-large-v3"
   
   # Option B : Créer manuellement les fichiers .txt
   # Pour chaque audio.wav, créer audio.txt avec la transcription
   ```

3. **Créer metadata.jsonl** :
   ```bash
   python -c "
   import json
   from pathlib import Path
   import librosa
   
   audio_dir = Path('benchmark/education/audio')
   refs_dir = Path('benchmark/education/refs')
   
   metadata = []
   for audio_file in sorted(audio_dir.glob('*.wav')):
       ref_file = refs_dir / f'{audio_file.stem}.txt'
       if ref_file.exists():
           duration = len(librosa.load(str(audio_file), sr=16000)[0]) / 16000
           metadata.append({
               'id': audio_file.stem,
               'source': 'manual',
               'audio_path': f'benchmark/education/audio/{audio_file.name}',
               'ref_path': f'benchmark/education/refs/{ref_file.name}',
               'duration': round(duration, 2),
               'sampling_rate': 16000
           })
   
   with open('benchmark/education/metadata.jsonl', 'w') as f:
       for m in metadata:
           f.write(json.dumps(m, ensure_ascii=False) + '\n')
   
   print(f'✅ Created metadata.jsonl with {len(metadata)} entries')
   "
   ```

## Option 2 : Utiliser un Serveur avec Plus d'Espace

1. **Sur un serveur cloud** (RunPod, Modal, etc.) :
   ```bash
   git clone https://github.com/mathisescriva/Experimentations_Gilbert-STT.git
   cd Experimentations_Gilbert-STT
   pip install -r requirements.txt
   python scripts/build_education_v1.py
   ```

2. **Télécharger le dataset créé** :
   ```bash
   # Depuis le serveur
   tar -czf education_v1.tar.gz benchmark/education/
   
   # Depuis votre Mac
   scp user@server:/path/to/education_v1.tar.gz ./
   tar -xzf education_v1.tar.gz
   ```

## Option 3 : Libérer de l'Espace d'Abord

```bash
# Nettoyer le cache HuggingFace (ATTENTION: re-téléchargera tout)
rm -rf ~/.cache/huggingface

# Nettoyer d'autres caches
pip cache purge
brew cleanup  # Si vous utilisez Homebrew

# Vérifier l'espace libéré
df -h ~
```

Puis relancer :
```bash
python scripts/build_education_v1.py --summre-limit 5 --voxpopuli-limit 5 --no-pastel-clone
```

## Option 4 : Dataset Minimal de Test

Créer juste 2-3 fichiers pour tester le pipeline :

```bash
# Créer 2 fichiers audio de test (si vous en avez)
# + leurs transcriptions
# + metadata.jsonl minimal

# Le benchmark fonctionnera avec même 1 fichier pour tester
```

