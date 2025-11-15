#!/bin/bash
# Transfert de fichiers via SSH en utilisant base64 (quand SCP ne fonctionne pas)

SSH_HOST="2qyiuevis8oycw-64410d88@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/Experimentations_Gilbert-STT"
LOCAL_DIR="/Users/mathisescriva/Desktop/Experimentations_Gilbert-STT"

echo "🚀 Transfert des fichiers via base64..."

# Fonction pour transférer un fichier
transfer_file() {
    local file=$1
    local remote_path=$2
    
    echo "📤 Transfert de $file..."
    
    # Encoder le fichier en base64 et le transférer
    cat "$file" | base64 | ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" \
        "mkdir -p $(dirname $remote_path) && base64 -d > $remote_path" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ $file transféré"
    else
        echo "❌ Erreur pour $file"
        return 1
    fi
}

# Créer le répertoire distant
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" "mkdir -p $REMOTE_DIR" 2>/dev/null

# Transférer les fichiers
cd "$LOCAL_DIR"

transfer_file "train_whisper_fr.py" "$REMOTE_DIR/train_whisper_fr.py"
transfer_file "requirements.txt" "$REMOTE_DIR/requirements.txt"
transfer_file "README.md" "$REMOTE_DIR/README.md"
transfer_file "inference_example.py" "$REMOTE_DIR/inference_example.py"

echo ""
echo "✅ Transfert terminé !"

