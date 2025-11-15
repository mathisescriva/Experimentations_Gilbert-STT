"""
Expérience 1 : Fine-tuning Whisper Large V3 sur Multilingual LibriSpeech (French)
Objectif : Créer gilbert-whisper-l3-fr-base-v1
VERSION B200 (Blackwell - Plus puissante mais plus chère)
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
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "huggingface-hub>=0.19.0",
        "hf_transfer",
        "torchcodec",  # Nécessaire pour décoder l'audio dans datasets
    )
)

app = modal.App("whisper-exp1-fr-base-b200")


@app.function(
    image=image,
    gpu="B200",  # B200 Blackwell - Plus puissante que H200 (~1.5x plus rapide)
    timeout=86400,  # 24 heures
    volumes={
        "/model_cache": modal.Volume.from_name("whisper-models", create_if_missing=True),
        "/output": modal.Volume.from_name("whisper-output", create_if_missing=True),
    },
)
def train_whisper():
    """Fonction d'entraînement - Expérience 1 avec B200"""
    import os
    import torch
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperProcessor,
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
        processor: WhisperProcessor

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

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

    def prepare_dataset(example, processor):
        """Prépare un exemple pour l'entraînement"""
        audio = example["audio"]
        
        # Extraire les features audio
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Tokenizer le texte
        labels = processor.tokenizer(example["text"]).input_ids
        
        # Gérer le cas où labels est une liste de listes
        if isinstance(labels[0], list):
            labels = labels[0]
        
        # Remplacer pad_token_id par -100 pour le loss
        labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "input_features": input_features.numpy(),
            "labels": labels
        }

    def compute_metrics(pred, processor, metric):
        """Calcule le WER"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Remplacer -100 par pad_token_id pour le décodage
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Décoder
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Normaliser
        normalizer = BasicTextNormalizer()
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]
        
        # Calculer WER
        wer = metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}

    # ========== MAIN TRAINING ==========
    print("=" * 60)
    print("🚀 EXPÉRIENCE 1 : Fine-tuning Whisper Large V3")
    print("📚 Dataset: Multilingual LibriSpeech (French)")
    print("🎯 Objectif: gilbert-whisper-l3-fr-base-v1")
    print("🔥 GPU: B200 (Blackwell - ULTIME)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device}")
    
    # Désactiver hf_transfer si problème
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    # Charger le modèle
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    
    # Configuration importante : désactiver forced_decoder_ids et suppress_tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    print("   ✓ Model loaded")
    
    # Charger et préparer les datasets
    datasets = load_and_prepare_datasets()
    
    # Préparer les datasets pour l'entraînement
    print("\n🔧 Preparing datasets for training...")
    
    def prepare_fn(example):
        return prepare_dataset(example, processor)
    
    # Optimisation : plus de processus pour le preprocessing
    train_dataset = datasets["train"].map(
        prepare_fn,
        remove_columns=datasets["train"].column_names,
        num_proc=8,  # Augmenté de 4 à 8 pour accélérer le preprocessing
    )
    
    test_dataset = datasets["test"].map(
        prepare_fn,
        remove_columns=datasets["test"].column_names,
        num_proc=8,  # Augmenté de 4 à 8 pour accélérer le preprocessing
    )
    
    print("   ✓ Datasets prepared")
    
    # Data collator et métrique
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    
    # Arguments d'entraînement - OPTIMISÉS POUR VITESSE MAXIMALE (B200)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,  # B200 a 192GB de mémoire, on peut augmenter encore plus
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=1,
        fp16=False,  # On utilise bf16 (optimal sur B200)
        dataloader_num_workers=16,  # Plus de workers pour B200
        dataloader_pin_memory=True,
        evaluation_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        push_to_hub=False,
        gradient_checkpointing=False,  # B200 a largement assez de mémoire (192GB)
        bf16=True,  # BF16 optimal sur B200
    )
    
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, wer_metric)
    
    # Pas besoin de gradient checkpointing avec B200 (192GB de mémoire)
    # model.gradient_checkpointing_enable()  # Désactivé pour B200
    
    # Créer le trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )
    
    # Lancer l'entraînement
    print("\n" + "=" * 60)
    print("🏋️  Starting training...")
    print("=" * 60)
    
    trainer.train()
    
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
    print("🚀 Lancement de l'Expérience 1 sur Modal (B200)...")
    print("📚 Dataset: Multilingual LibriSpeech (French)")
    print("🎯 Objectif: gilbert-whisper-l3-fr-base-v1")
    print("🔥 GPU: B200 (Blackwell - Plus puissante)")
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

