# Guide : Préparer les Références pour le Benchmark

Ce guide explique comment obtenir automatiquement les transcriptions de référence sans avoir à les transcrire manuellement.

## 🎯 Méthodes Disponibles

### 1. 📚 Depuis des Datasets HuggingFace (RECOMMANDÉ)

**Meilleure option** : Utiliser des datasets publics qui ont déjà des transcriptions de haute qualité.

#### Exemple : Multilingual LibriSpeech (Français)

```bash
python -m src.evaluation.prepare_references dataset \
  --dataset "facebook/multilingual_librispeech" \
  --config "french" \
  --audio-column "audio" \
  --text-column "transcript" \
  --subset "meetings" \
  --limit 100
```

Cela va :
- Télécharger le dataset depuis HuggingFace
- Extraire les fichiers audio et leurs transcriptions
- Les sauvegarder dans `benchmark/meetings/audio/` et `benchmark/meetings/refs/`

#### Autres Datasets Utiles

**Pour les réunions :**
```bash
# VoxPopuli (discours longs, proches de réunions)
python -m src.evaluation.prepare_references dataset \
  --dataset "facebook/voxpopuli" \
  --config "fr" \
  --subset "meetings"
```

**Pour les accents régionaux :**
```bash
# Common Voice (diversité d'accents)
python -m src.evaluation.prepare_references dataset \
  --dataset "mozilla-foundation/common_voice_17_0" \
  --config "fr" \
  --subset "accents"
```

### 2. 🎬 Depuis des Fichiers de Sous-titres

Si vous avez des fichiers audio avec des sous-titres (.srt ou .vtt) :

```bash
# Installer les dépendances pour les sous-titres
pip install pysrt webvtt

# Extraire les transcriptions
python -m src.evaluation.prepare_references subtitles \
  --subtitle-dir /path/to/subtitles \
  --audio-dir benchmark/meetings/audio \
  --output-refs-dir benchmark/meetings/refs \
  --format srt
```

**Format attendu :**
- Fichiers audio : `meeting_001.wav`
- Fichiers sous-titres : `meeting_001.srt` (même nom de base)

### 3. 🤖 Utiliser un Modèle de Référence (Pseudo-références)

⚠️ **ATTENTION** : Cette méthode génère des "pseudo-références" (pas de vraies références). 
Utilisez un modèle de très haute qualité (Whisper Large V3) et **vérifiez manuellement** les résultats.

```bash
# Générer des pseudo-références avec Whisper Large V3
python -m src.evaluation.prepare_references model \
  --audio-dir benchmark/meetings/audio \
  --output-refs-dir benchmark/meetings/refs \
  --model-name "openai/whisper-large-v3" \
  --device cuda
```

**Quand utiliser cette méthode :**
- Pour un setup rapide et tester le pipeline
- Si vous avez des fichiers audio sans transcriptions
- **MAIS** : Vérifiez et corrigez manuellement avant d'utiliser pour l'évaluation finale

### 4. 📄 Depuis des Fichiers Texte Existants

Si vous avez déjà des fichiers texte avec les transcriptions :

```bash
python -m src.evaluation.prepare_references text \
  --text-dir /path/to/text/files \
  --output-refs-dir benchmark/meetings/refs \
  --audio-dir benchmark/meetings/audio  # Optionnel : vérifie que l'audio existe
```

## 🚀 Workflow Recommandé

### Option A : Datasets Publics (Idéal)

1. **Choisir un dataset approprié** selon le type de contenu
2. **Extraire les données** :
   ```bash
   python -m src.evaluation.prepare_references dataset \
     --dataset "facebook/multilingual_librispeech" \
     --config "french" \
     --subset "meetings" \
     --limit 50
   ```
3. **Vérifier quelques échantillons** manuellement
4. **Lancer le benchmark** !

### Option B : Vos Propres Données

1. **Placer vos fichiers audio** dans `benchmark/{subset}/audio/`
2. **Générer des pseudo-références** :
   ```bash
   python -m src.evaluation.prepare_references model \
     --audio-dir benchmark/meetings/audio \
     --output-refs-dir benchmark/meetings/refs \
     --model-name "openai/whisper-large-v3"
   ```
3. **Vérifier et corriger** les transcriptions générées
4. **Lancer le benchmark**

### Option C : Sous-titres Existants

1. **Placer audio + sous-titres** dans des dossiers séparés
2. **Extraire les transcriptions** :
   ```bash
   python -m src.evaluation.prepare_references subtitles \
     --subtitle-dir /path/to/subtitles \
     --audio-dir benchmark/meetings/audio \
     --output-refs-dir benchmark/meetings/refs
   ```
3. **Vérifier** que les noms correspondent
4. **Lancer le benchmark**

## 📋 Checklist

Avant de lancer le benchmark, vérifiez :

- [ ] Les fichiers audio sont dans `benchmark/{subset}/audio/`
- [ ] Les fichiers de référence sont dans `benchmark/{subset}/refs/`
- [ ] Les noms correspondent (ex: `audio.wav` → `audio.txt`)
- [ ] Les références sont en UTF-8
- [ ] Vous avez vérifié quelques échantillons manuellement

## 💡 Astuces

1. **Commencez petit** : Testez avec 10-20 échantillons d'abord
2. **Vérifiez la qualité** : Regardez quelques transcriptions générées
3. **Mélangez les sources** : Utilisez différents datasets pour différents sous-ensembles
4. **Conservez les originaux** : Gardez une copie de vos données brutes

## 🔧 Dépendances Optionnelles

Pour les sous-titres :
```bash
pip install pysrt webvtt
```

Ces dépendances ne sont pas dans `requirements.txt` car elles sont optionnelles.

