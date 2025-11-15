#!/bin/bash
# Script à copier-coller dans le terminal RunPod pour créer train_whisper_fr.py

python3 << 'PYEOF'
import base64

# Contenu encodé en base64
encoded = """IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMw=="""  # Ceci sera remplacé par le vrai contenu

# Décoder et écrire
with open('/workspace/train_whisper_fr.py', 'wb') as f:
    f.write(base64.b64decode(encoded))

print('✅ train_whisper_fr.py créé')
PYEOF

chmod +x /workspace/train_whisper_fr.py
ls -lah /workspace/train_whisper_fr.py

