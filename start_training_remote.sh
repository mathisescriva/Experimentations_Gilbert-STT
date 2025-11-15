#!/bin/bash
# Script pour lancer l'entraînement sur le serveur distant

SSH_HOST="2qyiuevis8oycw-64410d88@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/Experimentations_Gilbert-STT"

echo "🚀 Lancement de l'entraînement sur RunPod..."
echo ""

# Créer un script de démarrage sur le serveur
cat << 'REMOTE_SCRIPT' | base64 | ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY" "$SSH_HOST" "cd $REMOTE_DIR && base64 -d > start_training.sh && chmod +x start_training.sh && echo 'Script créé'"
#!/bin/bash
cd /workspace/Experimentations_Gilbert-STT

echo "🚀 Démarrage de l'entraînement Whisper Large V3" > training_status.log
echo "================================================" >> training_status.log
date >> training_status.log
echo "" >> training_status.log

# Vérifier le GPU
echo "🎮 Vérification du GPU..." >> training_status.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> training_status.log 2>&1
echo "" >> training_status.log

# Installer les dépendances
echo "📦 Installation des dépendances..." >> training_status.log
pip install --upgrade pip >> training_status.log 2>&1
pip install -r requirements.txt >> training_status.log 2>&1
echo "✅ Dépendances installées" >> training_status.log
echo "" >> training_status.log

# Lancer l'entraînement en arrière-plan
echo "🏋️  Lancement de l'entraînement..." >> training_status.log
nohup python3 train_whisper_fr.py > training.log 2>&1 &
TRAIN_PID=$!
echo "PID: $TRAIN_PID" >> training_status.log
echo "Entraînement lancé en arrière-plan" >> training_status.log

# Attendre un peu et afficher les premières lignes
sleep 5
echo "" >> training_status.log
echo "Premières lignes du log:" >> training_status.log
head -30 training.log >> training_status.log 2>&1 || echo "Log pas encore créé" >> training_status.log
REMOTE_SCRIPT

# Exécuter le script
echo "📤 Exécution du script sur le serveur..."
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY" "$SSH_HOST" "cd $REMOTE_DIR && bash start_training.sh"

# Récupérer le statut
echo ""
echo "📊 Statut de l'entraînement:"
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$SSH_KEY" "$SSH_HOST" "cd $REMOTE_DIR && cat training_status.log 2>/dev/null || echo 'Statut pas encore disponible'"

echo ""
echo "✅ Entraînement lancé !"
echo ""
echo "📋 Pour voir les logs en temps réel:"
echo "   ssh $SSH_HOST -i $SSH_KEY 'cd $REMOTE_DIR && tail -f training.log'"
echo ""

