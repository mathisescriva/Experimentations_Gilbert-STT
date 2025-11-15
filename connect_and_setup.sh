#!/bin/bash
# Script pour se connecter à RunPod et configurer l'entraînement automatiquement

# Configuration
RUNPOD_SSH="2qyiuevis8oycw-64410d88@ssh.runpod.io"
RUNPOD_TCP="root@38.147.83.16"
RUNPOD_PORT="37674"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_DIR="/Users/mathisescriva/Desktop/Experimentations_Gilbert-STT"
REMOTE_DIR="/workspace/Experimentations_Gilbert-STT"

echo "🚀 Connexion à RunPod et configuration automatique"
echo "=================================================="

# Tester la connexion SSH
echo ""
echo "📡 Test de connexion SSH..."
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY" "$RUNPOD_SSH" "echo 'Connexion OK'" 2>/dev/null; then
    SSH_HOST="$RUNPOD_SSH"
    SSH_OPTS="-i $SSH_KEY"
    echo "✅ Connexion via ssh.runpod.io réussie"
elif ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$RUNPOD_PORT" -i "$SSH_KEY" "$RUNPOD_TCP" "echo 'Connexion OK'" 2>/dev/null; then
    SSH_HOST="$RUNPOD_TCP"
    SSH_OPTS="-p $RUNPOD_PORT -i $SSH_KEY"
    echo "✅ Connexion via TCP direct réussie"
else
    echo "❌ Impossible de se connecter via SSH"
    echo ""
    echo "Votre clé SSH n'est pas encore configurée dans RunPod."
    echo ""
    echo "Votre clé publique est :"
    cat "$SSH_KEY.pub"
    echo ""
    echo "📋 Pour configurer SSH :"
    echo "1. Allez sur RunPod.io → Votre profil → SSH Keys"
    echo "2. Ajoutez la clé publique ci-dessus"
    echo "3. Relancez ce script"
    echo ""
    echo "💡 Alternative : Utilisez Jupyter Lab (lien dans l'interface RunPod)"
    exit 1
fi

# Vérifier le GPU
echo ""
echo "🎮 Vérification du GPU..."
ssh $SSH_OPTS "$SSH_HOST" "nvidia-smi" || echo "⚠️  GPU non détecté"

# Créer le répertoire distant
echo ""
echo "📁 Création du répertoire de travail..."
ssh $SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_DIR"

# Transférer les fichiers
echo ""
echo "📤 Transfert des fichiers..."
scp $SSH_OPTS \
    "$LOCAL_DIR/train_whisper_fr.py" \
    "$LOCAL_DIR/requirements.txt" \
    "$LOCAL_DIR/README.md" \
    "$LOCAL_DIR/inference_example.py" \
    "$SSH_HOST:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ Fichiers transférés avec succès"
else
    echo "❌ Erreur lors du transfert des fichiers"
    exit 1
fi

# Installer les dépendances
echo ""
echo "📦 Installation des dépendances..."
ssh $SSH_OPTS "$SSH_HOST" "cd $REMOTE_DIR && pip install --upgrade pip && pip install -r requirements.txt"

if [ $? -eq 0 ]; then
    echo "✅ Dépendances installées"
else
    echo "⚠️  Erreur lors de l'installation (peut-être déjà installé)"
fi

# Afficher les informations finales
echo ""
echo "=================================================="
echo "✅ Configuration terminée !"
echo ""
echo "📋 Pour lancer l'entraînement, connectez-vous avec :"
echo "   ssh $SSH_OPTS $SSH_HOST"
echo ""
echo "   Puis exécutez :"
echo "   cd $REMOTE_DIR"
echo "   python train_whisper_fr.py"
echo ""
echo "💡 Ou lancez directement :"
echo "   ssh $SSH_OPTS $SSH_HOST 'cd $REMOTE_DIR && python train_whisper_fr.py'"
echo "=================================================="

