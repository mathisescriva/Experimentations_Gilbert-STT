# 🚀 Quick Start : Créer le Dataset education_v1

## Étape 1 : Installer les dépendances

```bash
cd /Users/mathisescriva/Desktop/Experimentations_Gilbert-STT
pip install librosa soundfile jsonlines
```

Ou réinstaller toutes les dépendances :
```bash
pip install -r requirements.txt
```

## Étape 2 : Lancer le script de construction

```bash
python scripts/build_education_v1.py
```

Cette commande va :
1. ✅ Télécharger SUMM-RE depuis HuggingFace (~15 fichiers)
2. ✅ Télécharger VoxPopuli FR depuis HuggingFace (~8 fichiers)  
3. ✅ Cloner PASTEL depuis GitHub (~15 fichiers, si git disponible)
4. ✅ Traiter et sauvegarder dans `benchmark/education/`

**Durée estimée : 15-25 minutes**

## Étape 3 : Vérifier le résultat

```bash
# Compter les fichiers créés
ls benchmark/education/audio/*.wav | wc -l
ls benchmark/education/refs/*.txt | wc -l

# Voir les métadonnées
head -3 benchmark/education/metadata.jsonl
```

## ⚠️ Si PASTEL ne fonctionne pas

Si vous voyez des erreurs avec PASTEL, vous pouvez l'ignorer :

```bash
python scripts/build_education_v1.py --no-pastel-clone
```

Cela créera le dataset avec seulement SUMM-RE et VoxPopuli (23 fichiers).

## 📊 Résultat attendu

Après exécution, vous devriez avoir :
- `benchmark/education/audio/*.wav` - Fichiers audio
- `benchmark/education/refs/*.txt` - Transcriptions
- `benchmark/education/metadata.jsonl` - Métadonnées

## 🎯 Utiliser le dataset

Une fois créé, le dataset est automatiquement disponible pour le benchmark (déjà ajouté dans `configs/benchmark.yaml`).

Lancer le benchmark :
```bash
python -m src.evaluation.run_benchmark --config configs/benchmark.yaml
```
