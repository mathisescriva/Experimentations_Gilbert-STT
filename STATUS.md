# 📊 Statut de l'entraînement

## ✅ Actions effectuées

1. ✅ **Connexion SSH configurée** - Clé SSH ajoutée dans RunPod
2. ✅ **Fichiers transférés** - Tous les fichiers sont sur le pod
3. ✅ **Script de lancement créé** - `launch_training.sh` prêt

## 🚀 Pour lancer l'entraînement

### Option 1 : Via le script automatique

```bash
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
./launch_training.sh
```

### Option 2 : Manuellement via SSH

```bash
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/Experimentations_Gilbert-STT
pip install -r requirements.txt
python train_whisper_fr.py
```

## 📊 Monitoring

### Vérifier que l'entraînement tourne

```bash
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/Experimentations_Gilbert-STT
ps aux | grep train_whisper
```

### Voir les logs en temps réel

```bash
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/Experimentations_Gilbert-STT
tail -f training.log
```

### Vérifier le GPU

```bash
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519
nvidia-smi
```

## 📁 Fichiers sur le pod

Tous les fichiers sont dans : `/workspace/Experimentations_Gilbert-STT/`

- `train_whisper_fr.py` - Script principal
- `requirements.txt` - Dépendances
- `README.md` - Documentation
- `inference_example.py` - Script d'inférence

## ⏱️ Temps estimé

- **Téléchargement datasets** : 30-60 minutes (première fois)
- **Téléchargement modèle** : 5-10 minutes
- **Entraînement 1 epoch** : 3-6 heures
- **Total** : ~4-8 heures

## 💰 Coût estimé

Avec RTX A6000 (~$0.79/h) : **~$3-6** pour un entraînement complet

## 🎯 Prochaines étapes

1. Lancer l'entraînement (si pas déjà fait)
2. Monitorer la progression
3. Récupérer le modèle finetuné une fois terminé

---

**Le modèle sera sauvegardé dans** : `/workspace/Experimentations_Gilbert-STT/gilbert-whisper-large-v3-fr-v1/`

