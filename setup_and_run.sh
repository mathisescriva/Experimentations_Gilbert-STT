#!/bin/bash
# Script à exécuter sur le serveur RunPod

cd /workspace/Experimentations_Gilbert-STT

echo "🚀 Démarrage de l'entraînement Whisper Large V3"
echo "================================================"
echo ""

# Vérifier le GPU
echo "🎮 Vérification du GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo ""

# Lancer l'entraînement
echo "🏋️  Lancement de l'entraînement..."
echo "================================================"
echo ""

python3 train_whisper_fr.py

echo ""
echo "✅ Entraînement terminé !"

