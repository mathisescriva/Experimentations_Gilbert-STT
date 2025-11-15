#!/usr/bin/env python3
"""
Script pour se connecter au pod RunPod et installer hf_transfer
"""

import paramiko
import sys
import os

# Configuration
SSH_HOST = "ssh.runpod.io"
SSH_USER = "2qyiuevis8oycw-64410d88"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")

def main():
    print("🔌 Connexion au pod RunPod...")
    
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
        print("✅ Connecté !\n")
        
        # Vérifier le GPU
        print("🎮 Vérification du GPU...")
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        gpu_info = stdout.read().decode().strip()
        if gpu_info:
            print(gpu_info)
        print()
        
        # Vérifier les fichiers
        print("📁 Vérification des fichiers...")
        stdin, stdout, stderr = ssh.exec_command("cd /workspace && ls -lah *.py *.txt 2>/dev/null || echo 'Pas de fichiers'")
        files = stdout.read().decode().strip()
        print(files)
        print()
        
        # Installer hf_transfer
        print("📦 Installation de hf_transfer...")
        stdin, stdout, stderr = ssh.exec_command("pip install hf_transfer")
        install_output = stdout.read().decode()
        install_errors = stderr.read().decode()
        if install_output:
            print(install_output)
        if install_errors and "error" not in install_errors.lower():
            print(install_errors)
        print("✅ hf_transfer installé (ou déjà installé)\n")
        
        # Alternative: désactiver hf_transfer
        print("🔧 Configuration alternative: désactiver hf_transfer...")
        stdin, stdout, stderr = ssh.exec_command(
            "cd /workspace && "
            "echo 'export HF_HUB_ENABLE_HF_TRANSFER=0' >> ~/.bashrc && "
            "echo 'Variable d\\'environnement configurée'"
        )
        config_output = stdout.read().decode()
        print(config_output)
        print()
        
        # Vérifier l'espace disque
        print("💾 Espace disque disponible...")
        stdin, stdout, stderr = ssh.exec_command("df -h /workspace | tail -1")
        disk_info = stdout.read().decode().strip()
        print(disk_info)
        print()
        
        print("✅ Configuration terminée !")
        print("\n📋 Prochaines étapes dans votre notebook:")
        print("   1. Ajoutez au début de votre code:")
        print("      import os")
        print("      os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'")
        print("   2. Ou installez hf_transfer dans une cellule:")
        print("      !pip install hf_transfer")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        ssh.close()

if __name__ == "__main__":
    main()

