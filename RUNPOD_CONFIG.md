# 🎯 Configuration RunPod pour Whisper Large V3

## ✅ Configuration RECOMMANDÉE (Meilleur rapport qualité/prix)

### Option 1 : RTX A6000 48GB (RECOMMANDÉ) ⭐

**Pourquoi** :
- ✅ 48GB VRAM (suffisant pour Whisper Large V3)
- ✅ Bon rapport performance/prix
- ✅ Stable et fiable

**Configuration** :
- **GPU** : RTX A6000 48GB
- **Template** : `PyTorch 2.0` ou `PyTorch 2.1`
- **OS** : Ubuntu 22.04
- **Disque** : 50GB minimum (100GB recommandé pour datasets)
- **RAM** : 32GB minimum
- **Prix** : ~$0.79/heure

**Commande pour lancer** :
```bash
# Après connexion au pod
pip install -r requirements.txt
python train_whisper_fr.py
```

---

### Option 2 : A100 40GB (Si disponible)

**Pourquoi** :
- ✅ GPU le plus puissant
- ✅ 40GB VRAM (largement suffisant)
- ✅ Entraînement plus rapide

**Configuration** :
- **GPU** : NVIDIA A100 40GB
- **Template** : `PyTorch 2.0` ou `PyTorch 2.1`
- **OS** : Ubuntu 22.04
- **Disque** : 100GB minimum (pour datasets + modèle)
- **RAM** : 64GB (généralement inclus)
- **Prix** : ~$1.10/heure

---

### Option 3 : A100 80GB (Si budget plus large)

**Pourquoi** :
- ✅ 80GB VRAM (très confortable)
- ✅ Permet des batch sizes plus grands
- ✅ Entraînement encore plus rapide

**Configuration** :
- **GPU** : NVIDIA A100 80GB
- **Template** : `PyTorch 2.0` ou `PyTorch 2.1`
- **OS** : Ubuntu 22.04
- **Disque** : 100GB minimum
- **Prix** : ~$1.50/heure

---

## ❌ Configurations à ÉVITER

### RTX 3090 / RTX 4090
- ⚠️ 24GB VRAM peut être limite pour Whisper Large V3
- ⚠️ Risque d'erreur OOM (Out Of Memory)

### GPU avec moins de 24GB VRAM
- ❌ Insuffisant pour Whisper Large V3
- ❌ Ne fonctionnera pas

---

## 📋 Étapes détaillées pour créer le pod

### 1. Aller sur RunPod.io
- Créer un compte / Se connecter
- Aller dans "Pods" → "Deploy"

### 2. Choisir le template
- **Template** : `RunPod PyTorch 2.0` ou `RunPod PyTorch 2.1`
- Ou chercher "PyTorch" dans les templates

### 3. Sélectionner le GPU
- **Recommandé** : RTX A6000 48GB
- **Alternative** : A100 40GB ou 80GB

### 4. Configurer le stockage
- **Disque système** : 50GB minimum
- **Volume persistant** (optionnel mais recommandé) : 100GB
  - Permet de garder les datasets entre les sessions
  - Évite de re-télécharger à chaque fois

### 5. Configuration réseau
- **Ports** : Ouvrir le port 6006 pour TensorBoard (optionnel)
- **Jupyter** : Activé si vous voulez utiliser Jupyter

### 6. Lancer le pod
- Cliquer sur "Deploy"
- Attendre 1-2 minutes que le pod démarre

---

## 🔧 Configuration après le démarrage

### 1. Se connecter au pod

**Via Terminal (SSH)** :
```bash
# RunPod vous donnera une commande SSH
ssh root@<pod-ip> -p <port>
```

**Via Jupyter** (si activé) :
- Ouvrir l'URL Jupyter fournie
- Ouvrir un terminal dans Jupyter

### 2. Vérifier le GPU

```bash
nvidia-smi
```

Vous devriez voir votre GPU (A6000 ou A100) avec la mémoire disponible.

### 3. Installer les dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt
```

### 4. Vérifier l'espace disque

```bash
df -h
```

Assurez-vous d'avoir au moins 50GB libres.

---

## 💰 Estimation des coûts

### Avec RTX A6000 48GB (~$0.79/h)

- **Téléchargement datasets** : 30-60 min = ~$0.40-0.80
- **Téléchargement modèle** : 5-10 min = ~$0.07-0.13
- **Entraînement 1 epoch** : 3-6 heures = ~$2.40-4.80
- **TOTAL** : ~$3-6 pour un entraînement complet

### Avec A100 40GB (~$1.10/h)

- **Téléchargement datasets** : 30-60 min = ~$0.55-1.10
- **Téléchargement modèle** : 5-10 min = ~$0.09-0.18
- **Entraînement 1 epoch** : 2-4 heures = ~$2.20-4.40
- **TOTAL** : ~$3-6 pour un entraînement complet

---

## 🎯 Recommandation finale

**Pour débuter** : **RTX A6000 48GB**
- ✅ Prix raisonnable
- ✅ Performance suffisante
- ✅ 48GB VRAM confortable

**Si budget plus large** : **A100 40GB**
- ✅ Plus rapide
- ✅ Meilleure stabilité

**Éviter** : GPU avec moins de 40GB VRAM pour Whisper Large V3

---

## 📝 Checklist avant de lancer

- [ ] Pod créé avec GPU A6000 ou A100
- [ ] Template PyTorch 2.0/2.1 sélectionné
- [ ] Disque 50GB+ configuré
- [ ] Volume persistant 100GB (optionnel mais recommandé)
- [ ] Pod démarré et accessible
- [ ] `nvidia-smi` fonctionne
- [ ] Code transféré sur le pod
- [ ] `pip install -r requirements.txt` exécuté
- [ ] Prêt à lancer `python train_whisper_fr.py`

---

## 🚀 Commandes rapides une fois connecté

```bash
# 1. Vérifier GPU
nvidia-smi

# 2. Aller dans le dossier du projet
cd /workspace/Experimentations_Gilbert-STT  # ou votre chemin

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Lancer l'entraînement
python train_whisper_fr.py

# 5. (Optionnel) Monitorer avec TensorBoard
tensorboard --logdir ./gilbert-whisper-large-v3-fr-v1/runs --port 6006
```

---

## 💡 Astuce : Volume persistant

Créez un **volume persistant** de 100GB pour :
- ✅ Stocker les datasets (évite de re-télécharger)
- ✅ Garder les checkpoints
- ✅ Sauvegarder le modèle finetuné

Cela vous fera économiser du temps et de l'argent sur les prochains entraînements !

---

Bon entraînement ! 🚀

