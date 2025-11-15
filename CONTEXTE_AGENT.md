# 🎯 CONTEXTE COMPLET - Fine-tuning Whisper Large V3 pour le Français

## 📋 OBJECTIF FINAL

Fine-tuner `openai/whisper-large-v3` pour améliorer les performances en français, en créant un pipeline d'expériences progressives :

1. **Expérience 1 (EN COURS)** : Base française propre → `gilbert-whisper-l3-fr-base-v1`
2. **Expérience 2** (futur) : Robustesse réunions → `gilbert-whisper-l3-fr-meetings-v1`
3. **Expérience 3** (futur) : Robustesse téléphone/bruit
4. **Expérience 4** (futur) : Accents régionaux

**Objectif global** : Améliorer la transcription en français tout en préservant les capacités multilingues.

---

## 🚀 EXPÉRIENCE 1 : Base française propre (EN COURS)

### Objectif
Créer `gilbert-whisper-l3-fr-base-v1` en fine-tunant sur du français propre, long, stable, non-bruyant.

### Dataset utilisé
- **Dataset** : `facebook/multilingual_librispeech`
- **Config** : `french`
- **Colonnes** :
  - Audio : `audio` (casté en 16kHz mono)
  - Texte : `transcript`
- **Split** : 95% train (245,302 échantillons) / 5% test (12,911 échantillons)

### Configuration d'entraînement
- **Modèle de base** : `openai/whisper-large-v3`
- **GPU** : H200 (141GB mémoire)
- **Batch size** : 24
- **Gradient accumulation** : 1
- **Learning rate** : 1e-5
- **Epochs** : 1
- **BF16** : True (optimal sur H200)
- **Gradient checkpointing** : False (assez de mémoire)
- **Group by length** : True (optimisation)
- **Evaluation** : Toutes les 5000 steps
- **Output** : `/output/gilbert-whisper-l3-fr-base-v1`

### Modifications importantes du modèle
```python
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
```

---

## 📁 FICHIERS IMPORTANTS

### Script principal
- **`train_whisper_exp1.py`** : Script d'entraînement pour l'Expérience 1
  - Utilise Modal pour l'exécution sur GPU H200
  - Mode détaché : `modal run --detach train_whisper_exp1.py`
  - Cache du preprocessing dans `/preprocessed_data` (Volume Modal)

### Scripts d'évaluation
- **`evaluate_exp1.py`** : Évaluation complète après entraînement
  - Compare baseline vs fine-tuné
  - Calcule WER
  - Test multilingue
  - Génère `evaluation_exp1_results.json`

### Documentation
- **`README.md`** : Documentation générale du projet
- **`PROCHAINES_ETAPES_EXP1.md`** : Guide des prochaines étapes après Expérience 1
- **`METRIQUES_EXP1.md`** : Guide d'interprétation des métriques

### Autres fichiers
- **`train_whisper_fr.py`** : Version locale (non utilisée actuellement)
- **`requirements.txt`** : Dépendances Python

---

## 🔧 PROBLÈMES RENCONTRÉS ET SOLUTIONS

### Problème 1 : Timeouts multiprocessing
**Symptôme** : `TimeoutError` lors du preprocessing avec `num_proc=4` ou `8`

**Solution** : Utiliser `batched=True` dans `.map()` (comme l'exemple officiel Modal)
```python
train_dataset = datasets["train"].map(
    prepare_fn,
    batched=True,  # CRUCIAL
    num_proc=4,
)
```

### Problème 2 : `jiwer` manquant
**Symptôme** : `ImportError: To be able to use evaluate-metric/wer, you need to install jiwer`

**Solution** : Ajouter `jiwer` dans les dépendances de l'image Modal

### Problème 3 : Déconnexion client
**Symptôme** : "Stopping app - local client disconnected"

**Solution** : Utiliser `modal run --detach` pour continuer même si le client se déconnecte

### Problème 4 : `prepare_dataset` version batchée
**Solution** : Changer de version "exemple par exemple" à version batchée
```python
# AVANT (ne fonctionnait pas bien)
def prepare_dataset(example, processor):
    ...

# APRÈS (fonctionne)
def prepare_dataset(batch, processor):
    audio_arrays = [item["array"] for item in batch["audio"]]
    inputs = processor.feature_extractor(audio_arrays, ...)
    batch["input_features"] = inputs.input_features
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    batch["input_length"] = [len(arr) for arr in audio_arrays]
    return batch
```

---

## 📊 ÉTAT ACTUEL (Nov 15, 16:50)

### Preprocessing en cours
- **Progression** : ~3% (7,000 / 245,302 échantillons)
- **Vitesse** : ~101 examples/s
- **Temps restant estimé** : ~39 minutes
- **Total preprocessing** : ~40 minutes

### Après le preprocessing
1. Sauvegarde dans `/preprocessed_data` (Volume Modal)
2. Démarrage automatique de l'entraînement
3. Temps d'entraînement estimé : ~3.3 heures (H200, batch 24)
4. Évaluations à steps 5000, 10000, 15000
5. Sauvegarde finale dans `/output/gilbert-whisper-l3-fr-base-v1`

### Temps total estimé
- Preprocessing : ~40 minutes (en cours)
- Entraînement : ~3.3 heures
- **Total** : ~4 heures

---

## 🎯 MÉTRIQUES ATTENDUES

### Objectifs de l'Expérience 1
1. **WER amélioré** : WER fine-tuné < WER baseline
2. **Multilingue préservé** : Le modèle peut toujours transcrire d'autres langues
3. **Qualité FR améliorée** : Meilleure transcription sur français

### WER (Word Error Rate)
- **Excellent** : < 0.10 (10%)
- **Bon** : 0.10-0.20 (10-20%)
- **Acceptable** : 0.20-0.30 (20-30%)
- **À améliorer** : > 0.30 (30%)

### Amélioration attendue
- **Objectif minimum** : WER fine-tuné < WER baseline
- **Objectif idéal** : Réduction de 5-15% du WER
- **Excellente amélioration** : Réduction de 15%+

---

## 🔄 WORKFLOW COMPLET

### 1. Entraînement (EN COURS)
```bash
modal run --detach train_whisper_exp1.py
```

### 2. Après l'entraînement
1. Vérifier que le modèle est sauvegardé dans `/output/gilbert-whisper-l3-fr-base-v1`
2. Télécharger le modèle depuis Modal Volume
3. Lancer l'évaluation : `python evaluate_exp1.py`
4. Vérifier les métriques dans `evaluation_exp1_results.json`

### 3. Prochaines expériences
- **Expérience 2** : Fine-tuner `gilbert-whisper-l3-fr-base-v1` sur `facebook/voxpopuli` (FR) pour robustesse réunions
- **Expérience 3** : Fine-tuner sur données bruitées pour robustesse téléphone
- **Expérience 4** : Fine-tuner sur `common_voice` (FR) pour accents régionaux

---

## ⚙️ CONFIGURATION TECHNIQUE

### Plateforme
- **Modal** : Infrastructure cloud avec GPU H200
- **Volumes Modal** :
  - `/model_cache` : Cache des modèles HuggingFace
  - `/output` : Modèles fine-tunés sauvegardés
  - `/preprocessed_data` : Cache du preprocessing

### Dépendances clés
- `torch>=2.0.0`
- `transformers>=4.36.0`
- `datasets>=2.14.0`
- `accelerate>=0.25.0`
- `jiwer` (pour métrique WER)
- `torchcodec` (pour décoder audio)
- `ffmpeg` (pour torchcodec)

### Optimisations appliquées
- **GPU H200** : Plus rapide que A100 (~3.4x)
- **Batch size 24** : Optimal pour H200 (141GB mémoire)
- **BF16** : Plus rapide que FP16 sur H200
- **Group by length** : Optimise l'entraînement
- **batched=True** : Évite les timeouts multiprocessing

---

## 📝 NOTES IMPORTANTES

### Points critiques
1. **Toujours utiliser `batched=True`** dans `.map()` pour éviter les timeouts
2. **Utiliser `--detach`** pour que l'entraînement continue même si le client se déconnecte
3. **Le preprocessing est sauvegardé** dans `/preprocessed_data` pour éviter de le refaire
4. **Chaque expérience fine-tune le modèle précédent**, pas le modèle original

### Commandes utiles
```bash
# Lancer l'entraînement (détaché)
modal run --detach train_whisper_exp1.py

# Vérifier les apps en cours
modal app list

# Voir les logs
modal app logs <app-id>

# Arrêter une app
modal app stop <app-id>
```

### Structure des modèles
```
gilbert-whisper-l3-fr-base-v1      (Expérience 1 - EN COURS)
    ↓
gilbert-whisper-l3-fr-meetings-v1 (Expérience 2 - futur)
    ↓
gilbert-whisper-l3-fr-robust-v1   (Expérience 3 - futur)
    ↓
gilbert-whisper-l3-fr-final-v1    (Expérience 4 - futur)
```

---

## 🎯 PROCHAINES ACTIONS IMMÉDIATES

1. **Attendre la fin du preprocessing** (~40 minutes restantes)
2. **Vérifier que l'entraînement démarre** automatiquement
3. **Suivre la progression** sur Modal dashboard
4. **Après l'entraînement** : Télécharger le modèle et l'évaluer

---

## 💡 CONTEXTE ADDITIONNEL

### Pourquoi cette approche
- Fine-tuning progressif pour ne pas casser les capacités existantes
- Chaque expérience améliore un aspect spécifique
- Le modèle de base (Expérience 1) sert de socle pour les suivantes

### Limitations locales (Mac M3 Pro)
- Pas d'entraînement complet en local (trop lourd)
- Utilisation de Modal pour GPU cloud
- Développement/test du code en local OK

### Coûts
- H200 : Plus cher à l'heure mais ~3.4x plus rapide
- Coût total similaire à A100 mais gain de temps énorme
- Temps total estimé : ~4 heures pour Expérience 1

---

**Dernière mise à jour** : Nov 15, 2025 - 16:50 CET
**Statut** : Preprocessing en cours (~3%, ~39 min restantes)

