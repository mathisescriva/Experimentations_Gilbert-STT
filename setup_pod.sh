#!/bin/bash
# Script à exécuter sur le pod RunPod

cd /workspace

echo "=== Installation hf_transfer ==="
pip install hf_transfer 2>&1

echo ""
echo "=== Fichiers présents ==="
ls -lah *.py *.txt 2>&1

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1

echo ""
echo "=== Configuration environnement ==="
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "HF_HUB_ENABLE_HF_TRANSFER=0" >> ~/.bashrc
echo "Variable configurée"

echo ""
echo "=== Espace disque ==="
df -h /workspace | tail -1

echo ""
echo "=== Vérification Python ==="
python3 --version
which python3

echo ""
echo "✅ Configuration terminée !"

