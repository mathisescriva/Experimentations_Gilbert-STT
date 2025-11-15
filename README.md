# Fine-tuning Whisper Large V3 pour le Français

Ce projet permet de fine-tuner le modèle `openai/whisper-large-v3` pour améliorer les performances en français et en contexte de réunions/discussions longues.

## 🎯 Objectif

Améliorer progressivement Whisper Large V3 pour :
- ✅ Améliorer la transcription en français
- ✅ Améliorer la robustesse en réunions/discussions longues
- ✅ Conserver les capacités multilingues

## 📚 Datasets utilisés

Le script utilise actuellement des datasets publics :

1. **fsicoli/common_voice_17_0** (FR)
   - Diversité d'accents et conditions réelles
   - Colonne texte : `sentence`

2. **facebook/voxpopuli** (FR)
   - Discours longs, oratoires, proches de réunions
   - Colonne texte : `normalized_text` ou `raw_text` (détection automatique)

3. **diarizers-community/ami** (optionnel, pour plus tard)
   - Corpus de réunions (anglais), speech spontané & overlaps
   - À intégrer plus tard pour "apprendre le style réunion"

## 📦 Installation

### Prérequis

- Python 3.8+
- CUDA (pour GPU cloud) ou Metal (pour Mac M3 Pro)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### 🚫 Limitations sur MacBook Pro M3 Pro

**À NE PAS faire en local :**
- ❌ Fine-tuner Whisper Large V3 complet → trop lourd, GPU Metal insuffisant
- ❌ Entraînement multi-epoch sur datasets FR complets → trop lent

**Ce qui est faisable en local :**
- ✅ Développement et test du code
- ✅ Téléchargement/preprocessing des datasets
- ✅ Vérification du pipeline avec un mini-training (quelques batches)
- ✅ Fine-tuning léger/LoRA sur Whisper-small/medium si besoin
- ✅ Tests d'inférence

### 🖥️ Développement local (Mac M3 Pro)

#### 1. Tester le chargement des datasets

Vous pouvez modifier temporairement le script pour tester avec un sous-échantillon :

```python
# Dans train_whisper_fr.py, ajouter après le chargement :
dataset = dataset.select(range(100))  # Test avec 100 échantillons
```

#### 2. Mini-training de test

Pour vérifier que le pipeline fonctionne, vous pouvez réduire les paramètres :

```bash
# Modifier dans train_whisper_fr.py :
# - num_train_epochs = 0.1  # Juste quelques batches
# - max_steps = 10  # Limiter à 10 steps
# - per_device_train_batch_size = 1  # Batch size minimal

python train_whisper_fr.py
```

⚠️ **Note** : Même avec ces réductions, le training complet sera très lent sur Mac M3 Pro. Utilisez cela uniquement pour valider que le code fonctionne.

### ☁️ Training sur GPU Cloud (A100)

#### Option 1 : RunPod

1. **Créer une instance RunPod**
   - Template : PyTorch
   - GPU : A100 40GB ou 80GB
   - OS : Ubuntu 22.04

2. **Se connecter et cloner le projet**

```bash
# Sur votre Mac
scp -r . user@runpod-ip:/workspace/whisper-finetuning/

# Ou cloner depuis Git
git clone <votre-repo> /workspace/whisper-finetuning
cd /workspace/whisper-finetuning
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Lancer l'entraînement**

```bash
# Avec accelerate (recommandé pour multi-GPU)
accelerate config  # Configurer une fois
accelerate launch train_whisper_fr.py

# Ou directement
python train_whisper_fr.py
```

#### Option 2 : Lambda Labs

1. **Créer une instance Lambda Labs**
   - GPU : A100 40GB ou 80GB

2. **Même procédure que RunPod**

#### Option 3 : HuggingFace Spaces / Inference Endpoints

Pour un setup plus simple, vous pouvez utiliser les ressources HuggingFace.

### 📥 Récupérer les poids finetunés

Une fois l'entraînement terminé sur le cloud :

```bash
# Depuis le serveur cloud
# Option 1 : Télécharger via SCP
scp -r user@cloud-ip:/workspace/whisper-finetuning/gilbert-whisper-large-v3-fr-v1 ./

# Option 2 : Upload vers HuggingFace Hub (si configuré)
# Dans train_whisper_fr.py, activer push_to_hub=True
```

## 🔧 Configuration

### Paramètres d'entraînement

Les paramètres par défaut dans `train_whisper_fr.py` :

- `learning_rate`: 1e-5
- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `num_train_epochs`: 1 (phase test)
- `fp16`: True (mixed precision)
- `predict_with_generate`: True

### Modifier les paramètres

Éditez directement `train_whisper_fr.py` ou utilisez des variables d'environnement :

```bash
export LEARNING_RATE=1e-5
export BATCH_SIZE=4
python train_whisper_fr.py
```

## 🧪 Inférence avec le modèle finetuné

### Utilisation du script d'exemple

Un script d'inférence est fourni pour faciliter la transcription :

```bash
python inference_example.py path/to/audio.wav
```

Ou avec un modèle personnalisé :

```bash
python inference_example.py path/to/audio.wav --model-path ./gilbert-whisper-large-v3-fr-v1
```

### Utilisation en Python

Vous pouvez aussi utiliser le modèle directement dans votre code :

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Charger le modèle finetuné
processor = WhisperProcessor.from_pretrained("./gilbert-whisper-large-v3-fr-v1")
model = WhisperForConditionalGeneration.from_pretrained("./gilbert-whisper-large-v3-fr-v1")

# Charger l'audio
audio_path = "path/to/audio.wav"
audio, sr = torchaudio.load(audio_path)
audio = audio.squeeze().numpy()

# Prétraiter
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Générer
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])

# Décoder
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

## 📊 Monitoring

Le script utilise TensorBoard pour le monitoring. Pour visualiser :

```bash
tensorboard --logdir ./gilbert-whisper-large-v3-fr-v1/runs
```

## 🐛 Dépannage

### Erreur de mémoire GPU

Réduire `per_device_train_batch_size` ou augmenter `gradient_accumulation_steps` :

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

### Datasets trop volumineux

Utiliser le streaming :

```python
dataset = load_dataset(..., streaming=True)
```

### Problèmes de téléchargement

Les datasets sont téléchargés automatiquement au premier lancement. Vérifiez votre connexion internet et l'espace disque disponible.

## 📊 Benchmark

Ce projet inclut un système de benchmark interne pour évaluer les performances ASR sur différents types de contenu.

### Structure du benchmark

Le benchmark est organisé en sous-ensembles dans le dossier `benchmark/` :

```
benchmark/
├── meetings/
│   ├── audio/     # Fichiers audio de réunions
│   └── refs/      # Textes de référence (.txt)
├── telephone/
│   ├── audio/     # Fichiers audio téléphoniques
│   └── refs/
├── accents/
│   ├── audio/      # Fichiers avec accents régionaux
│   └── refs/
└── longform/
    ├── audio/      # Contenu long format
    └── refs/
```

Chaque fichier audio doit avoir un fichier de référence correspondant dans `refs/` avec le même nom (ex: `audio/sample.wav` → `refs/sample.txt`).

### Utilisation

1. **Préparer les données** : Placez vos fichiers audio et leurs références dans les dossiers appropriés.

2. **Configurer le benchmark** : Éditez `configs/benchmark.yaml` :
   ```yaml
   model_name: "MEscriva/gilbert-fr-source"  # Votre modèle
   device: "cuda"  # ou "cpu"
   subsets: ["meetings", "telephone", "accents", "longform"]
   ```

3. **Lancer le benchmark** :
   ```bash
   python -m src.evaluation.run_benchmark --config configs/benchmark.yaml
   ```

   Ou avec des options personnalisées :
   ```bash
   python -m src.evaluation.run_benchmark \
     --config configs/benchmark.yaml \
     --model-name "MEscriva/gilbert-fr-source" \
     --compute-cer \
     --output-csv results.csv
   ```

### Métriques

Le benchmark calcule actuellement :
- **WER (Word Error Rate)** : Taux d'erreur au niveau des mots
- **CER (Character Error Rate)** : Taux d'erreur au niveau des caractères (optionnel avec `--compute-cer`)

Les résultats sont affichés dans un tableau récapitulatif par sous-ensemble et une moyenne globale.

## 📝 Structure du projet

```
.
├── train_whisper_exp1.py      # Script principal d'entraînement (Modal)
├── evaluate_exp1.py            # Script d'évaluation après entraînement
├── inference_example.py        # Script d'exemple pour l'inférence
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
├── .gitignore                  # Fichiers à ignorer par Git
├── configs/
│   └── benchmark.yaml         # Configuration du benchmark
├── src/
│   └── evaluation/
│       ├── run_benchmark.py    # Script CLI pour le benchmark
│       ├── asr_inference.py    # Wrapper pour l'inférence ASR
│       └── metrics.py           # Calcul des métriques (WER, CER)
├── benchmark/                  # Données de benchmark
│   ├── meetings/
│   ├── telephone/
│   ├── accents/
│   └── longform/
├── data/                       # (optionnel) Données locales
└── models/                     # (optionnel) Checkpoints
    └── gilbert-whisper-l3-fr-base-v1/  # Modèle finetuné
```

## 🔮 Prochaines étapes

- [ ] Intégrer le dataset AMI pour les réunions
- [ ] Implémenter LoRA pour un fine-tuning plus efficace
- [ ] Ajouter des métriques supplémentaires (CER, BLEU)
- [ ] Support multi-GPU avec DeepSpeed
- [ ] Script d'évaluation dédié

## 📄 Licence

Ce projet utilise des modèles et datasets sous leurs licences respectives. Vérifiez les licences avant utilisation commerciale.

## 🙏 Remerciements

- OpenAI pour le modèle Whisper
- HuggingFace pour les outils transformers
- Les contributeurs des datasets publics utilisés

