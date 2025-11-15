# 🚀 Prochaines Étapes - Guide d'Action

## ✅ Étape 1 : Vérifier que le test a fonctionné

Vérifiez si le dossier de test a été créé :

```bash
ls -la gilbert-whisper-large-v3-fr-v1-test/
```

Si le dossier existe avec des fichiers, le test a réussi ! ✅

---

## 🖥️ Étape 2 : Choisir votre environnement d'entraînement

### Option A : Entraînement sur GPU Cloud (RECOMMANDÉ) ⭐

Pour un entraînement complet et efficace, utilisez un GPU cloud :

#### 2.1. Choisir un fournisseur

**RunPod** (recommandé pour débuter)
- Site : https://www.runpod.io
- GPU A100 40GB : ~$1.10/heure
- Template PyTorch disponible

**Lambda Labs**
- Site : https://lambdalabs.com
- GPU A100 : ~$1.10/heure

**HuggingFace Spaces** (si vous avez un compte Pro)
- Plus simple mais moins flexible

#### 2.2. Préparer votre projet pour le cloud

**Méthode 1 : Via Git (recommandé)**

```bash
# Sur votre Mac, initialiser Git si pas déjà fait
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
git init
git add train_whisper_fr.py requirements.txt README.md inference_example.py
git commit -m "Initial commit - Whisper fine-tuning"
git remote add origin <VOTRE_REPO_GIT>
git push -u origin main
```

Puis sur le serveur cloud :
```bash
git clone <VOTRE_REPO_GIT>
cd Experimentations_Gilbert-STT
```

**Méthode 2 : Via SCP (transfert direct)**

```bash
# Depuis votre Mac
scp -r /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT user@cloud-ip:/workspace/
```

#### 2.3. Sur le serveur cloud

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer accelerate (optionnel, pour multi-GPU)
accelerate config

# 3. Lancer l'entraînement
python train_whisper_fr.py

# OU avec accelerate
accelerate launch train_whisper_fr.py
```

#### 2.4. Récupérer le modèle finetuné

```bash
# Depuis votre Mac, télécharger le modèle
scp -r user@cloud-ip:/workspace/Experimentations_Gilbert-STT/gilbert-whisper-large-v3-fr-v1 ./
```

---

### Option B : Test local approfondi (Mac M3 Pro) ⚠️

**ATTENTION** : L'entraînement complet sera très lent (plusieurs jours).

Si vous voulez quand même tester localement :

```bash
# Modifier train_whisper_fr.py pour réduire encore plus :
# - MAX_SAMPLES = 50 (au lieu de tout le dataset)
# - num_train_epochs = 0.01
# - max_steps = 2

python train_whisper_fr.py
```

---

## 📊 Étape 3 : Monitorer l'entraînement

### Sur GPU Cloud

Le script génère automatiquement des logs TensorBoard :

```bash
# Sur le serveur cloud, dans un autre terminal
tensorboard --logdir ./gilbert-whisper-large-v3-fr-v1/runs --port 6006

# Puis accéder via : http://cloud-ip:6006
```

### Métriques à surveiller

- **Loss** : doit diminuer progressivement
- **WER (Word Error Rate)** : doit diminuer (meilleur = plus bas)
- **Temps par epoch** : pour estimer la durée totale

---

## 🧪 Étape 4 : Tester le modèle finetuné

Une fois l'entraînement terminé :

```bash
# Utiliser le script d'inférence
python inference_example.py path/to/votre/audio.wav

# Ou avec le modèle finetuné spécifique
python inference_example.py path/to/votre/audio.wav --model-path ./gilbert-whisper-large-v3-fr-v1
```

---

## 🔧 Étape 5 : Ajuster les hyperparamètres (si nécessaire)

Si les résultats ne sont pas satisfaisants, vous pouvez modifier dans `train_whisper_fr.py` :

```python
# Augmenter le learning rate
learning_rate=2e-5  # au lieu de 1e-5

# Augmenter les epochs
num_train_epochs=3  # au lieu de 1

# Ajuster le batch size selon votre GPU
per_device_train_batch_size=2  # si mémoire insuffisante
gradient_accumulation_steps=8  # compenser le batch size réduit
```

---

## 📝 Checklist avant de lancer l'entraînement complet

- [ ] Le test local a fonctionné (dossier de test créé)
- [ ] Serveur GPU cloud configuré (RunPod/Lambda/etc.)
- [ ] Code transféré sur le serveur cloud
- [ ] Dépendances installées sur le cloud
- [ ] Espace disque suffisant (au moins 100GB pour datasets + modèle)
- [ ] Budget estimé (A100 ~$1.10/h, entraînement ~3-6h = $3-7)

---

## 🆘 En cas de problème

### Erreur de mémoire GPU

Réduire le batch size dans `train_whisper_fr.py` :
```python
per_device_train_batch_size=2  # au lieu de 4
gradient_accumulation_steps=8   # au lieu de 4
```

### Datasets ne se téléchargent pas

Vérifier la connexion internet et l'espace disque :
```bash
df -h  # Vérifier l'espace disque
```

### Le modèle ne s'améliore pas

- Augmenter le nombre d'epochs
- Vérifier la qualité des datasets
- Ajuster le learning rate

---

## 🎯 Résumé rapide

1. **Maintenant** : Vérifier que le test a fonctionné
2. **Ensuite** : Configurer un serveur GPU cloud (RunPod recommandé)
3. **Puis** : Transférer le code et lancer `train_whisper_fr.py`
4. **Enfin** : Récupérer le modèle et tester avec `inference_example.py`

---

## 💡 Astuce

Pour économiser du temps et de l'argent, vous pouvez :
- Commencer avec 1 epoch pour valider le pipeline
- Puis relancer avec 3 epochs une fois que tout fonctionne
- Sauvegarder les checkpoints régulièrement

Bon entraînement ! 🚀

