# Guide : Construire le Dataset education_v1

Ce guide explique comment générer le dataset `education_v1` à partir de zéro.

## 📋 Prérequis

### 1. Installer les dépendances

```bash
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
pip install -r requirements.txt
```

Dépendances nécessaires :
- `datasets` (pour HuggingFace)
- `librosa` (pour le traitement audio)
- `soundfile` (pour sauvegarder les fichiers audio)
- `jsonlines` (pour metadata.jsonl)
- `tqdm` (barre de progression)

### 2. Vérifier que git est installé (pour PASTEL)

```bash
git --version
```

Si git n'est pas installé, le script fonctionnera quand même mais ne pourra pas cloner PASTEL automatiquement.

## 🚀 Lancer la construction

### Option 1 : Construction complète (recommandé)

```bash
python scripts/build_education_v1.py
```

Cette commande va :
1. ✅ Télécharger SUMM-RE depuis HuggingFace (15 fichiers)
2. ✅ Télécharger VoxPopuli FR depuis HuggingFace (8 fichiers)
3. ✅ Cloner PASTEL depuis GitHub (15 fichiers) - si git est disponible
4. ✅ Traiter tous les fichiers audio et transcriptions
5. ✅ Générer `metadata.jsonl`

### Option 2 : Sans PASTEL (si problème avec git)

```bash
python scripts/build_education_v1.py --no-pastel-clone
```

Cela construira le dataset avec seulement SUMM-RE et VoxPopuli (23 fichiers au lieu de 38).

### Option 3 : Personnaliser les limites

```bash
python scripts/build_education_v1.py \
  --summre-limit 20 \
  --voxpopuli-limit 10 \
  --pastel-limit 20
```

### Option 4 : Avec PASTEL déjà cloné localement

Si vous avez déjà cloné le repo PASTEL :

```bash
git clone https://github.com/nicolashernandez/anr-pastel-data.git data/pastel
python scripts/build_education_v1.py --pastel-dir data/pastel
```

## 📊 Ce qui va se passer

### Étape 1 : Téléchargement des données

Le script va :
- Télécharger les datasets depuis HuggingFace (peut prendre quelques minutes)
- Cloner le repo PASTEL depuis GitHub (si activé)
- Afficher la progression avec des barres de progression

### Étape 2 : Traitement

Pour chaque fichier :
- ✅ Conversion audio en 16kHz mono WAV
- ✅ Normalisation du texte (lowercase, suppression ponctuation)
- ✅ Validation de la durée (10s - 10min)
- ✅ Sauvegarde dans `benchmark/education/audio/` et `refs/`

### Étape 3 : Génération des métadonnées

Création de `benchmark/education/metadata.jsonl` avec :
- ID, source, chemins, durée, sampling rate

### Étape 4 : Résumé final

Le script affiche :
- Nombre de fichiers par source
- Durée totale par source
- Durée totale du dataset

## 📁 Structure finale

Après exécution, vous aurez :

```
benchmark/education/
├── audio/
│   ├── summre_01.wav
│   ├── summre_02.wav
│   ├── voxp_01.wav
│   ├── pastel_01.wav
│   └── ...
├── refs/
│   ├── summre_01.txt
│   ├── summre_02.txt
│   ├── voxp_01.txt
│   ├── pastel_01.txt
│   └── ...
└── metadata.jsonl
```

## ⚠️ Notes importantes

### PASTEL et fichiers audio

Le corpus PASTEL contient des **transcriptions** mais les fichiers audio peuvent manquer car ils doivent être extraits depuis les vidéos sources (COCo, Canal-U).

Si vous voyez :
```
⚠️  No audio found for <file>.stm, skipping
```

Cela signifie que la transcription existe mais pas l'audio. Dans ce cas :
- Le dataset sera construit avec SUMM-RE et VoxPopuli uniquement
- Ou vous devrez extraire l'audio depuis les vidéos sources

### Durée estimée

- Téléchargement SUMM-RE : ~2-5 minutes
- Téléchargement VoxPopuli : ~2-5 minutes
- Clonage PASTEL : ~1-2 minutes
- Traitement : ~5-10 minutes
- **Total : ~15-25 minutes**

### Espace disque nécessaire

- Datasets HuggingFace (cache) : ~500MB-1GB
- PASTEL (clone) : ~50-100MB
- Dataset final : ~50-200MB
- **Total estimé : ~1-2GB**

## 🐛 Dépannage

### Erreur : "ModuleNotFoundError"

```bash
pip install datasets librosa soundfile jsonlines tqdm
```

### Erreur : "git not found"

Installez git ou utilisez `--no-pastel-clone` pour ignorer PASTEL.

### Erreur : Timeout lors du téléchargement

Les datasets HuggingFace peuvent être lents. Réessayez simplement :
```bash
python scripts/build_education_v1.py
```

### Pas de fichiers PASTEL

Si PASTEL ne charge aucun fichier :
1. Vérifiez que git est installé
2. Vérifiez votre connexion internet
3. Ou clonez manuellement : `git clone https://github.com/nicolashernandez/anr-pastel-data.git data/pastel`

## ✅ Vérification

Après exécution, vérifiez :

```bash
# Compter les fichiers
ls benchmark/education/audio/*.wav | wc -l
ls benchmark/education/refs/*.txt | wc -l

# Vérifier metadata.jsonl
head -5 benchmark/education/metadata.jsonl
```

Vous devriez voir ~15-38 fichiers selon les sources disponibles.

## 🎯 Utilisation du dataset

Une fois construit, vous pouvez l'utiliser avec le benchmark :

```bash
# Mettre à jour configs/benchmark.yaml
# Ajouter "education" dans la liste des subsets

python -m src.evaluation.run_benchmark --config configs/benchmark.yaml
```

