# 🔧 Résoudre les Problèmes d'Espace Disque

Si vous voyez l'erreur "No space left on device", voici comment résoudre :

## 🧹 Solution 1 : Nettoyer le Cache HuggingFace

```bash
# Voir la taille du cache
du -sh ~/.cache/huggingface

# Nettoyer les datasets (garde les modèles)
rm -rf ~/.cache/huggingface/datasets

# Ou nettoyer tout le cache (ATTENTION: re-téléchargera tout)
rm -rf ~/.cache/huggingface
```

## 🧹 Solution 2 : Nettoyer d'autres caches

```bash
# Cache pip
pip cache purge

# Cache Python
find ~/Library/Caches -name "__pycache__" -type d -exec rm -r {} + 2>/dev/null
```

## 💾 Solution 3 : Utiliser un Dataset Plus Petit

Le script a été modifié pour utiliser `multilingual_librispeech` en fallback si SUMM-RE échoue (plus petit, peut être déjà en cache).

## 🚀 Solution 4 : Lancer avec Moins de Fichiers

```bash
# Réduire les limites pour utiliser moins d'espace
python scripts/build_education_v1.py \
  --summre-limit 5 \
  --voxpopuli-limit 5 \
  --pastel-limit 5 \
  --no-pastel-clone
```

Cela créera un dataset minimal avec seulement 10 fichiers.

## 📊 Vérifier l'Espace Disponible

```bash
df -h ~
```

Si vous avez moins de 2GB libres, nettoyez d'abord le cache.

