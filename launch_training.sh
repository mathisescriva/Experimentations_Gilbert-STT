#!/bin/bash
# Script pour lancer l'entraînement sur RunPod

SSH_HOST="2qyiuevis8oycw-64410d88@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/Experimentations_Gilbert-STT"

echo "🚀 Lancement de l'entraînement Whisper sur RunPod..."
echo "=================================================="

# Vérifier le GPU
echo ""
echo "🎮 Vérification du GPU..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1 | grep -v "PTY" || echo "GPU détecté"

# Installer les dépendances
echo ""
echo "📦 Installation des dépendances..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" \
    "cd $REMOTE_DIR && pip install --upgrade pip > /dev/null 2>&1 && pip install -r requirements.txt > /tmp/install.log 2>&1 && echo '✅ Dépendances installées' || (echo '⚠️  Vérification des dépendances...' && tail -5 /tmp/install.log)" 2>&1 | grep -v "PTY"

# Lancer l'entraînement
echo ""
echo "🏋️  Lancement de l'entraînement..."
echo "=================================================="
echo ""
echo "💡 L'entraînement va commencer. Vous pouvez :"
echo "   - Laisser tourner en arrière-plan"
echo "   - Surveiller les logs"
echo "   - Le processus peut prendre plusieurs heures"
echo ""
echo "📊 Pour monitorer, connectez-vous et lancez :"
echo "   ssh $SSH_HOST -i $SSH_KEY"
echo "   tail -f $REMOTE_DIR/gilbert-whisper-large-v3-fr-v1/training.log"
echo ""
echo "=================================================="
echo ""

# Lancer l'entraînement (en arrière-plan avec nohup pour qu'il continue même si la connexion se coupe)
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" \
    "cd $REMOTE_DIR && nohup python train_whisper_fr.py > training.log 2>&1 & echo 'Entraînement lancé en arrière-plan (PID: \$!)' && sleep 2 && tail -20 training.log" 2>&1 | grep -v "PTY"

echo ""
echo "✅ Entraînement lancé !"
echo ""
echo "📋 Pour vérifier le statut :"
echo "   ssh $SSH_HOST -i $SSH_KEY 'cd $REMOTE_DIR && tail -f training.log'"
echo ""

