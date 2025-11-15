# 🚀 Configuration Modal pour l'entraînement Whisper

## 📋 Étapes d'installation

### 1. Installer Modal

```bash
pip install modal
```

### 2. Configurer Modal (première fois)

```bash
python3 -m modal setup
```

Cette commande va :
- Ouvrir votre navigateur pour vous authentifier
- Créer un token API
- Configurer Modal sur votre machine

### 3. Lancer l'entraînement

```bash
modal run train_whisper_modal.py
```

## 🎯 Avantages de Modal

- ✅ **Pas de gestion de serveurs** - Modal gère tout automatiquement
- ✅ **GPU A100 automatique** - Accès direct aux GPU puissants
- ✅ **Pas de problèmes SSH** - Tout se fait via l'API Modal
- ✅ **Volumes persistants** - Les modèles sont sauvegardés automatiquement
- ✅ **Monitoring** - Suivez la progression via l'interface web Modal

## 📊 Monitoring

Pendant l'exécution, vous pouvez :
- Voir les logs en temps réel dans le terminal
- Accéder à l'interface web Modal pour plus de détails
- Les modèles sont sauvegardés dans un volume Modal persistant

## 💰 Coût

Modal facture à l'utilisation :
- GPU A100 : ~$1.10/heure
- Vous payez uniquement pendant l'exécution

## 📁 Récupérer le modèle

Une fois l'entraînement terminé, le modèle est dans le volume Modal. Vous pouvez :
1. Le télécharger via l'interface web Modal
2. Utiliser l'API Modal pour le récupérer
3. Le garder dans le volume pour les prochains entraînements

## 🔧 Personnalisation

Vous pouvez modifier dans `train_whisper_modal.py` :
- `gpu="A100"` → changer le type de GPU
- `timeout=86400` → ajuster le timeout
- Les paramètres d'entraînement (batch size, learning rate, etc.)

## 🆘 En cas de problème

- Vérifiez que Modal est bien configuré : `modal token show`
- Consultez les logs : `modal app logs whisper-finetuning-fr`
- Interface web : https://modal.com/apps

