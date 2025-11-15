# ⚡ Instructions Rapides - Connexion RunPod

## 🔑 Étape 1 : Configurer SSH (2 minutes)

Votre clé publique SSH :
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGxVabCuHmUMI4m7qqe3HMPcwjQq4MdL6zbjn35Nr0Cp mathisescriva@MacBook-Pro-de-Mathis.local
```

**Action** :
1. Allez sur **RunPod.io** → Votre profil (icône en haut à droite)
2. Cliquez sur **"SSH Keys"**
3. Cliquez sur **"Add SSH Key"**
4. **Collez la clé ci-dessus**
5. **Sauvegardez**

---

## 🚀 Étape 2 : Lancer le script automatique

Une fois la clé SSH ajoutée, exécutez :

```bash
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
./connect_and_setup.sh
```

Ce script va automatiquement :
- ✅ Se connecter au pod
- ✅ Vérifier le GPU
- ✅ Transférer tous vos fichiers
- ✅ Installer les dépendances
- ✅ Vous donner les commandes pour lancer l'entraînement

---

## 🎯 Étape 3 : Lancer l'entraînement

Après le script, lancez :

```bash
ssh 2qyiuevis8oycw-64410d88@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/Experimentations_Gilbert-STT
python train_whisper_fr.py
```

---

## 💡 Alternative : Jupyter Lab (Sans SSH)

Si vous préférez ne pas configurer SSH :

1. **Cliquez sur "Port 8888 → Jupyter Lab"** dans l'interface RunPod
2. **Ouvrez un terminal** dans Jupyter
3. **Uploadez vos fichiers** via l'interface (bouton Upload)
4. **Exécutez** :
   ```bash
   pip install -r requirements.txt
   python train_whisper_fr.py
   ```

---

## 📋 Checklist

- [ ] Clé SSH ajoutée dans RunPod
- [ ] Script `connect_and_setup.sh` exécuté
- [ ] Fichiers transférés
- [ ] Dépendances installées
- [ ] Entraînement lancé

---

**Une fois la clé SSH ajoutée, dites-moi et je lancerai le script automatiquement !** 🚀

