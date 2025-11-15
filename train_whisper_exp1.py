"""
Expérience 1 : Fine-tuning Whisper Large V3 sur Multilingual LibriSpeech (French)
Objectif : Créer gilbert-whisper-l3-fr-base-v1
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # Nécessaire pour torchcodec
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "evaluate>=0.4.0",
        "jiwer>=3.0.0",  # Nécessaire pour la métrique WER - VERSION EXPLICITE
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "huggingface-hub>=0.19.0",
        "hf_transfer",
        "torchcodec",  # Nécessaire pour décoder l'audio dans datasets
    )
    .run_commands("pip install --upgrade jiwer")  # Force l'installation de jiwer
)

app = modal.App("whisper-exp1-fr-base")


@app.function(
    image=image,
    gpu="H200",  # H200 est plus rapide, remis comme demandé
    timeout=86400,  # 24 heures
    volumes={
        "/model_cache": modal.Volume.from_name("whisper-models", create_if_missing=True),
        "/output": modal.Volume.from_name("whisper-output", create_if_missing=True),
        "/preprocessed_data": modal.Volume.from_name("whisper-preprocessed", create_if_missing=True),
    },
)
def train_whisper():
    """Fonction d'entraînement - Expérience 1"""
    import os
    import torch
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperProcessor,
        WhisperFeatureExtractor,  # Ajouté pour charger séparément
        WhisperTokenizer,  # Ajouté pour charger séparément
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    import evaluate
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import numpy as np

    # Configuration Expérience 1
    MODEL_NAME = "openai/whisper-large-v3"
    OUTPUT_DIR = "/output/gilbert-whisper-l3-fr-base-v1"
    SAMPLING_RATE = 16000
    TRAIN_TEST_SPLIT = 0.95

    # Dataset : Multilingual LibriSpeech French uniquement
    DATASET_NAME = "facebook/multilingual_librispeech"
    DATASET_CONFIG = "french"
    TEXT_COLUMN = "transcript"  # La colonne s'appelle "transcript" dans ce dataset
    AUDIO_COLUMN = "audio"

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Data collator that pads audio features and text labels for batch training.
        Basé sur l'exemple officiel Modal.
        """
        processor: WhisperProcessor
        decoder_start_token_id: int  # Ajouté comme dans l'exemple officiel

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            # Séparer les features audio et les labels texte (besoin de padding différent)
            model_input_name = self.processor.model_input_names[0]  # "input_features" pour Whisper
            input_features = [
                {model_input_name: feature[model_input_name]} for feature in features
            ]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            # Pad les features audio
            batch = self.processor.feature_extractor.pad(
                input_features,
                return_tensors="pt",
                return_attention_mask=True,  # Ajouté comme dans l'exemple officiel
                padding=True,  # Ajouté comme dans l'exemple officiel
            )

            # Pad les labels texte
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            
            # Remplacer les tokens de padding par -100 pour qu'ils soient ignorés dans le calcul de loss
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # IMPORTANT: Retirer le start token si le tokenizer l'a ajouté
            # Le modèle l'ajoutera automatiquement pendant l'entraînement
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    def load_and_prepare_datasets():
        """Charge Multilingual LibriSpeech French"""
        print("=" * 60)
        print("📚 Dataset: Multilingual LibriSpeech (French)")
        print("=" * 60)
        
        print(f"\n📦 Loading {DATASET_NAME} (config: {DATASET_CONFIG})...")
        
        try:
            # Charger le dataset
            dataset = load_dataset(
                DATASET_NAME,
                DATASET_CONFIG,
                split="train",
            )
            
            print(f"   ✓ Loaded: {len(dataset)} samples")
            print(f"   📋 Columns: {dataset.column_names}")
            
            # Vérifier les colonnes
            if AUDIO_COLUMN not in dataset.column_names:
                raise ValueError(f"Audio column '{AUDIO_COLUMN}' not found in dataset")
            if TEXT_COLUMN not in dataset.column_names:
                raise ValueError(f"Text column '{TEXT_COLUMN}' not found in dataset")
            
            # Caster l'audio en 16kHz mono
            print(f"\n🎵 Casting audio to {SAMPLING_RATE}Hz mono...")
            dataset = dataset.cast_column(
                AUDIO_COLUMN,
                Audio(sampling_rate=SAMPLING_RATE)
            )
            
            # Sélectionner et renommer les colonnes
            dataset = dataset.select_columns([AUDIO_COLUMN, TEXT_COLUMN])
            dataset = dataset.rename_columns({
                AUDIO_COLUMN: "audio",
                TEXT_COLUMN: "text"
            })
            
            # Filtrer les textes vides
            dataset = dataset.filter(
                lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 0
            )
            
            print(f"   ✓ Final dataset size: {len(dataset)} samples")
            
            # Shuffle et split
            dataset = dataset.shuffle(seed=42)
            split_dataset = dataset.train_test_split(
                test_size=1 - TRAIN_TEST_SPLIT,
                seed=42
            )
            
            print(f"\n📊 Train/Test split:")
            print(f"   ✓ Train: {len(split_dataset['train'])} samples")
            print(f"   ✓ Test: {len(split_dataset['test'])} samples")
            
            return split_dataset
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def prepare_dataset(batch, feature_extractor, tokenizer, model_input_name):
        """Prépare un batch d'exemples pour l'entraînement (BATCHED VERSION)"""
        # Version batchée comme dans l'exemple officiel Modal
        # IMPORTANT: Utiliser feature_extractor et tokenizer séparément (pas processor)
        audio_arrays = [item["array"] for item in batch["audio"]]
        
        # Extraire les features audio en batch
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
        )
        batch[model_input_name] = inputs.get(model_input_name)  # Utiliser model_input_name dynamique
        
        # Tokenizer les textes en batch
        batch["labels"] = tokenizer(batch["text"]).input_ids
        
        # Calculer la longueur pour group_by_length
        batch["input_length"] = [len(arr) for arr in audio_arrays]
        
        return batch

    def compute_metrics(pred, tokenizer, normalizer, metric):
        """Calcule le WER - comme l'exemple officiel Modal"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Remplacer -100 par pad_token_id pour le décodage
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        # Décoder
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Normaliser
        pred_str = [normalizer(pred).strip() for pred in pred_str]
        label_str = [normalizer(label).strip() for label in label_str]
        
        # Calculer WER
        wer = metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}

    # ========== MAIN TRAINING ==========
    print("=" * 60)
    print("🚀 EXPÉRIENCE 1 : Fine-tuning Whisper Large V3")
    print("📚 Dataset: Multilingual LibriSpeech (French)")
    print("🎯 Objectif: gilbert-whisper-l3-fr-base-v1")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device}")
    
    # Désactiver hf_transfer si problème
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    # Charger le modèle - IMPORTANT: Charger feature_extractor et tokenizer SÉPARÉMENT (comme l'exemple officiel)
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    
    # Configuration importante : désactiver forced_decoder_ids et suppress_tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    print("   ✓ Model loaded")
    
    # Créer le processor APRÈS (comme l'exemple officiel)
    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    
    # Charger et préparer les datasets
    datasets = load_and_prepare_datasets()
    
    # Préparer les datasets pour l'entraînement
    print("\n🔧 Preparing datasets for training...")
    
    # Vérifier si le preprocessing est déjà sauvegardé
    preprocessed_train_path = "/preprocessed_data/train_dataset"
    preprocessed_test_path = "/preprocessed_data/test_dataset"
    
    import os
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_test_path):
        print("   📦 Chargement du preprocessing sauvegardé...")
        from datasets import load_from_disk
        train_dataset = load_from_disk(preprocessed_train_path)
        test_dataset = load_from_disk(preprocessed_test_path)
        print("   ✓ Datasets préprocessés chargés depuis le cache")
    else:
        print("   🔄 Preprocessing des datasets (première fois)...")
        
        # Utiliser batched=True comme dans l'exemple officiel Modal
        # IMPORTANT: Passer feature_extractor et tokenizer séparément (pas processor)
        import functools
        import os
        model_input_name = feature_extractor.model_input_names[0]  # "input_features" pour Whisper
        prepare_fn = functools.partial(
            prepare_dataset,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            model_input_name=model_input_name,
        )
        
        train_dataset = datasets["train"].map(
            prepare_fn,
            batched=True,  # CRUCIAL: traite par batch au lieu d'un par un
            remove_columns=datasets["train"].column_names,
            num_proc=os.cpu_count(),  # Comme l'exemple officiel
            desc="Feature extract + tokenize (train)",
        )
        
        test_dataset = datasets["test"].map(
            prepare_fn,
            batched=True,  # CRUCIAL: traite par batch au lieu d'un par un
            remove_columns=datasets["test"].column_names,
            num_proc=os.cpu_count(),  # Comme l'exemple officiel
            desc="Feature extract + tokenize (test)",
        )
        
        # Sauvegarder pour la prochaine fois
        print("   💾 Sauvegarde du preprocessing...")
        train_dataset.save_to_disk(preprocessed_train_path)
        test_dataset.save_to_disk(preprocessed_test_path)
        print("   ✓ Datasets préparés et sauvegardés")
    
    # Data collator et métrique
    # IMPORTANT: Passer decoder_start_token_id comme dans l'exemple officiel Modal
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # Vérifier que jiwer est installé avant de charger la métrique
    try:
        import jiwer
        print("   ✓ jiwer installé")
    except ImportError:
        print("   ⚠️  jiwer non trouvé, installation...")
        import subprocess
        subprocess.check_call(["pip", "install", "jiwer"])
        import jiwer
    
    wer_metric = evaluate.load("wer")
    
    # Arguments d'entraînement - OPTIMISÉS POUR VITESSE MAXIMALE (H200)
    # Basé sur l'exemple officiel Modal
    # Gestion de la compatibilité eval_strategy vs evaluation_strategy
    import transformers
    transformers_version = transformers.__version__
    print(f"   📦 Transformers version: {transformers_version}")
    
    # Déterminer quel paramètre utiliser selon la version
    # Dans transformers >= 4.37, eval_strategy remplace evaluation_strategy
    try:
        # Essayer d'abord avec eval_strategy (versions récentes)
        training_args_dict = {
            "output_dir": OUTPUT_DIR,
            "per_device_train_batch_size": 24,  # H200 a 141GB de mémoire
            "per_device_eval_batch_size": 24,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "fp16": True,  # Comme l'exemple officiel Modal (bf16 est mieux pour H100/H200 mais fp16 est plus compatible)
            # Pas de dataloader_num_workers explicite (comme l'exemple officiel Modal)
            # "dataloader_pin_memory": True,  # Retiré pour correspondre à l'exemple officiel
            "eval_strategy": "steps",  # Paramètre moderne (transformers >= 4.37)
            "eval_steps": 5000,
            "save_strategy": "steps",
            "save_steps": 5000,
            "logging_steps": 100,
            "report_to": "tensorboard",
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "predict_with_generate": True,
            "generation_max_length": 225,
            "generation_num_beams": 1,  # Comme dans l'exemple officiel
            "save_total_limit": 3,
            "push_to_hub": False,
            "gradient_checkpointing": False,  # H200 a largement assez de mémoire
            # "bf16": True,  # Désactivé - utiliser fp16 comme l'exemple officiel (plus compatible)
            "group_by_length": True,  # Réactivé - comme l'exemple officiel (avec toutes les corrections, ça devrait fonctionner)
            "length_column_name": "input_length",  # Colonne créée dans prepare_dataset (pas utilisée si group_by_length=False)
        }
        training_args = Seq2SeqTrainingArguments(**training_args_dict)
    except TypeError as e:
        if "eval_strategy" in str(e):
            # Fallback vers evaluation_strategy pour versions anciennes
            print("   ⚠️  eval_strategy non supporté, utilisation de evaluation_strategy")
            training_args_dict["evaluation_strategy"] = training_args_dict.pop("eval_strategy")
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
        else:
            raise
    
    # Créer normalizer comme dans l'exemple officiel
    normalizer = BasicTextNormalizer()
    
    def compute_metrics_fn(pred):
        return compute_metrics(pred, tokenizer, normalizer, wer_metric)
    
    # Pas besoin de gradient checkpointing avec H200 (141GB de mémoire)
    # model.gradient_checkpointing_enable()  # Désactivé pour H200
    
    # Créer le trainer
    # IMPORTANT: Utiliser processing_class au lieu de tokenizer (déprécié)
    # Comme dans l'exemple officiel Modal
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=feature_extractor,  # IMPORTANT: Utiliser feature_extractor directement (pas processor.feature_extractor)
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    # Vérifier s'il existe un checkpoint pour reprendre l'entraînement
    import glob
    checkpoint_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
    resume_from_checkpoint = None
    
    if checkpoint_dirs:
        # Prendre le dernier checkpoint (le plus récent)
        resume_from_checkpoint = checkpoint_dirs[-1]
        print(f"\n🔄 Checkpoint trouvé : {resume_from_checkpoint}")
        print("   Reprise de l'entraînement depuis le checkpoint...")
    else:
        print("\n🆕 Aucun checkpoint trouvé, démarrage depuis le début")
    
    # Lancer l'entraînement
    print("\n" + "=" * 60)
    print("🏋️  Starting training...")
    print("=" * 60)
    
    # Logs de debug avant l'entraînement
    print(f"   📊 Train dataset size: {len(train_dataset)}")
    print(f"   📊 Eval dataset size: {len(test_dataset)}")
    print(f"   🎯 Output dir: {OUTPUT_DIR}")
    print(f"   💾 GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   🎮 GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("   ⏳ Calling trainer.train()...")
    import sys
    sys.stdout.flush()  # Force l'affichage
    
    # Ajouter un timeout et des logs supplémentaires
    print("   🔄 Initialisation du DataLoader et chargement du premier batch...")
    sys.stdout.flush()
    
    # IMPORTANT: Faire une évaluation baseline AVANT l'entraînement (comme l'exemple officiel Modal)
    # Cela initialise le trainer et peut éviter les blocages
    print("\n📊 Running baseline evaluation (initializes trainer)...")
    sys.stdout.flush()
    try:
        baseline_metrics = trainer.evaluate(
            metric_key_prefix="baseline",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        trainer.log_metrics("baseline", baseline_metrics)
        trainer.save_metrics("baseline", baseline_metrics)
        print(f"   ✓ Baseline WER: {baseline_metrics.get('baseline_wer', 'N/A')}")
        sys.stdout.flush()
    except Exception as e:
        print(f"   ⚠️  Baseline eval failed (continuing anyway): {e}")
        sys.stdout.flush()
    
    # Maintenant lancer l'entraînement
    if resume_from_checkpoint:
        print(f"\n🔄 Resuming from: {resume_from_checkpoint}")
        sys.stdout.flush()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("\n🆕 Starting training from scratch...")
        sys.stdout.flush()
        print(f"   📁 Weights will be saved to '{training_args.output_dir}'")
        sys.stdout.flush()
        trainer.train()
    
    print("   ✅ trainer.train() completed!")
    
    # Sauvegarder le modèle
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training completed!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print(f"🎯 Model name: gilbert-whisper-l3-fr-base-v1")
    
    # Évaluation finale
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Final WER: {eval_results.get('eval_wer', 'N/A')}")
    
    return {
        "wer": eval_results.get('eval_wer', 'N/A'),
        "output_dir": OUTPUT_DIR,
        "model_name": "gilbert-whisper-l3-fr-base-v1"
    }


@app.local_entrypoint()
def main():
    """Point d'entrée local"""
    print("🚀 Lancement de l'Expérience 1 sur Modal...")
    print("📚 Dataset: Multilingual LibriSpeech (French)")
    print("🎯 Objectif: gilbert-whisper-l3-fr-base-v1")
    print("=" * 60)
    
    try:
        result = train_whisper.remote()
        print(f"\n✅ Entraînement terminé !")
        print(f"📊 WER final: {result['wer']}")
        print(f"📁 Modèle sauvegardé dans: {result['output_dir']}")
        print(f"🎯 Nom du modèle: {result['model_name']}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

