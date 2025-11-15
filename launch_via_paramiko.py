#!/usr/bin/env python3
"""
Script pour lancer l'entraînement sur RunPod via SSH avec paramiko
"""

import paramiko
import sys
import time
import os

# Configuration
SSH_HOST = "ssh.runpod.io"
SSH_USER = "2qyiuevis8oycw-64410d88"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_DIR = "/workspace/Experimentations_Gilbert-STT"

def main():
    print("🚀 Connexion à RunPod et lancement de l'entraînement...")
    print("=" * 60)
    
    # Créer le client SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Charger la clé privée
        private_key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY_PATH)
        
        # Se connecter
        print(f"📡 Connexion à {SSH_USER}@{SSH_HOST}...")
        ssh.connect(
            hostname=SSH_HOST,
            username=SSH_USER,
            pkey=private_key,
            timeout=10
        )
        print("✅ Connecté !")
        print()
        
        # Vérifier le GPU
        print("🎮 Vérification du GPU...")
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        gpu_info = stdout.read().decode().strip()
        if gpu_info:
            print(gpu_info)
        else:
            print("⚠️  GPU non détecté ou erreur")
        print()
        
        # Installer les dépendances
        print("📦 Installation des dépendances...")
        stdin, stdout, stderr = ssh.exec_command(
            f"cd {REMOTE_DIR} && "
            "pip install --upgrade pip > /dev/null 2>&1 && "
            "pip install -r requirements.txt 2>&1 | tail -10"
        )
        output = stdout.read().decode()
        errors = stderr.read().decode()
        if output:
            print(output)
        if errors and "error" in errors.lower():
            print(f"⚠️  Erreurs: {errors}")
        print("✅ Dépendances installées (ou déjà installées)")
        print()
        
        # Lancer l'entraînement en arrière-plan
        print("🏋️  Lancement de l'entraînement...")
        print("=" * 60)
        
        # Créer un script de démarrage
        start_script = f"""#!/bin/bash
cd {REMOTE_DIR}
nohup python3 train_whisper_fr.py > training.log 2>&1 &
echo $! > training.pid
echo "Entraînement lancé (PID: $(cat training.pid))"
sleep 3
tail -30 training.log
"""
        
        # Transférer et exécuter le script
        stdin, stdout, stderr = ssh.exec_command(
            f"cd {REMOTE_DIR} && "
            f"cat > start_training.sh << 'EOF'\n{start_script}EOF\n"
            "chmod +x start_training.sh && "
            "bash start_training.sh"
        )
        
        output = stdout.read().decode()
        errors = stderr.read().decode()
        
        print(output)
        if errors:
            print(f"Erreurs: {errors}")
        
        print()
        print("=" * 60)
        print("✅ Entraînement lancé en arrière-plan !")
        print()
        print("📊 Pour voir les logs:")
        print(f"   ssh {SSH_USER}@{SSH_HOST} -i {SSH_KEY_PATH}")
        print(f"   cd {REMOTE_DIR} && tail -f training.log")
        print()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
    finally:
        ssh.close()

if __name__ == "__main__":
    # Vérifier si paramiko est installé
    try:
        import paramiko
    except ImportError:
        print("❌ paramiko n'est pas installé")
        print("📦 Installation: pip install paramiko")
        sys.exit(1)
    
    main()

