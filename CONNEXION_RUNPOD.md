# 🔌 Connexion à votre Pod RunPod

Votre pod est prêt ! Voici comment vous connecter et transférer votre code.

## 🎯 Option 1 : Via Jupyter Lab (LE PLUS SIMPLE) ⭐

### Étape 1 : Accéder à Jupyter
1. Dans l'interface RunPod, cliquez sur le lien **"Port 8888 → Jupyter Lab"**
2. Ou copiez l'URL et ouvrez-la dans votre navigateur
3. Vous devriez voir l'interface Jupyter Lab

### Étape 2 : Transférer vos fichiers

**Méthode A - Upload direct dans Jupyter** :
1. Dans Jupyter, cliquez sur "Upload" (icône flèche vers le haut)
2. Sélectionnez ces fichiers depuis votre Mac :
   - `train_whisper_fr.py`
   - `requirements.txt`
   - `README.md`
   - `inference_example.py`
3. Attendez que les uploads se terminent

**Méthode B - Via terminal dans Jupyter** :
1. Ouvrez un terminal dans Jupyter (New → Terminal)
2. Utilisez `wget` ou `curl` si vous avez mis les fichiers sur un serveur
3. Ou utilisez Git si vous avez créé un repo

### Étape 3 : Installer et lancer

Dans le terminal Jupyter :
```bash
# Vérifier le GPU
nvidia-smi

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'entraînement
python train_whisper_fr.py
```

---

## 🎯 Option 2 : Via SSH (Pour utilisateurs avancés)

### Étape 1 : Générer une clé SSH (si pas déjà fait)

Sur votre Mac :
```bash
# Vérifier si vous avez déjà une clé
ls -la ~/.ssh/id_ed25519

# Si pas de clé, en créer une
ssh-keygen -t ed25519 -C "your_email@example.com"
# Appuyez sur Entrée pour accepter l'emplacement par défaut
# Optionnel : ajouter un mot de passe
```

### Étape 2 : Ajouter la clé à RunPod

1. Dans RunPod, allez dans votre profil → SSH Keys
2. Copiez le contenu de votre clé publique :
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Collez-la dans RunPod

### Étape 3 : Se connecter via SSH

Dans votre terminal Mac :
```bash
# Utiliser la commande fournie par RunPod
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519

# OU via TCP direct (si vous préférez)
ssh root@38.147.83.16 -p 37674 -i ~/.ssh/id_ed25519
```

### Étape 4 : Transférer vos fichiers

**Méthode A - Via SCP (depuis votre Mac)** :
```bash
# Depuis votre Mac, dans un nouveau terminal
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT

# Transférer les fichiers
scp -P 37674 -i ~/.ssh/id_ed25519 \
  train_whisper_fr.py \
  requirements.txt \
  README.md \
  inference_example.py \
  root@38.147.83.16:/workspace/
```

**Méthode B - Via Git (recommandé)** :
```bash
# Sur votre Mac, initialiser Git si pas déjà fait
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
git init
git add train_whisper_fr.py requirements.txt README.md inference_example.py
git commit -m "Whisper fine-tuning"

# Créer un repo sur GitHub (ou utiliser un existant)
# Puis sur le pod RunPod :
cd /workspace
git clone <VOTRE_REPO_URL>
cd Experimentations_Gilbert-STT
```

### Étape 5 : Installer et lancer

Une fois connecté via SSH :
```bash
# Vérifier le GPU
nvidia-smi

# Aller dans le dossier
cd /workspace/Experimentations_Gilbert-STT

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'entraînement
python train_whisper_fr.py
```

---

## 🎯 Option 3 : Via Web Terminal (Si activé)

1. Activez le toggle "Enable Web Terminal" dans l'interface RunPod
2. Attendez que le terminal se charge
3. Utilisez-le comme un terminal normal
4. Suivez les mêmes étapes que pour SSH

---

## 📋 Checklist de connexion

- [ ] Pod est "Ready" (vert) ✅
- [ ] Méthode de connexion choisie (Jupyter recommandé)
- [ ] Fichiers transférés sur le pod
- [ ] GPU vérifié avec `nvidia-smi`
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Prêt à lancer `python train_whisper_fr.py`

---

## 🚀 Commandes rapides une fois connecté

```bash
# 1. Vérifier le GPU (doit montrer A6000 ou A100)
nvidia-smi

# 2. Vérifier l'espace disque
df -h

# 3. Aller dans le dossier du projet
cd /workspace/Experimentations_Gilbert-STT

# 4. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 5. Lancer l'entraînement
python train_whisper_fr.py

# 6. (Optionnel) Monitorer avec TensorBoard
tensorboard --logdir ./gilbert-whisper-large-v3-fr-v1/runs --port 6006
```

---

## 💡 Astuce : Volume persistant pour les datasets

Si vous avez créé un volume persistant :
```bash
# Monter le volume (si pas déjà monté)
# RunPod le monte généralement automatiquement dans /workspace

# Les datasets seront téléchargés dans :
# ~/.cache/huggingface/hub/

# Vous pouvez les déplacer sur le volume persistant pour les garder :
mkdir -p /workspace/datasets_cache
ln -s /workspace/datasets_cache ~/.cache/huggingface
```

---

## 🆘 En cas de problème

### "Permission denied" sur SSH
- Vérifiez que votre clé SSH est bien ajoutée dans RunPod
- Vérifiez les permissions : `chmod 600 ~/.ssh/id_ed25519`

### "No space left on device"
- Vérifiez l'espace : `df -h`
- Nettoyez le cache : `pip cache purge`
- Utilisez un volume persistant plus grand

### GPU non détecté
- Vérifiez que le pod utilise bien un GPU
- Redémarrez le pod si nécessaire

---

## 📊 Monitoring de l'entraînement

### Via TensorBoard (dans Jupyter)
1. Ouvrez un nouveau terminal dans Jupyter
2. Lancez : `tensorboard --logdir ./gilbert-whisper-large-v3-fr-v1/runs --port 6006`
3. Cliquez sur le lien TensorBoard qui apparaît

### Via les logs
```bash
# Suivre les logs en temps réel
tail -f ./gilbert-whisper-large-v3-fr-v1/training.log
```

---

## 🎯 Prochaines étapes

1. **Maintenant** : Connectez-vous via Jupyter (le plus simple)
2. **Ensuite** : Transférez vos fichiers
3. **Puis** : Installez les dépendances et lancez l'entraînement
4. **Enfin** : Surveillez la progression et récupérez le modèle finetuné

Bon entraînement ! 🚀

