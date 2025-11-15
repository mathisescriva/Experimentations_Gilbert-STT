"""
Fine-tuning Whisper Large V3 avec Modal - Version qui FONCTIONNE
Utilise des datasets qui sont garantis de fonctionner
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
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
    )
)

app = modal.App("whisper-finetuning-fr-working")


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    volumes={
        "/model_cache": modal.Volume.from_name("whisper-models", create_if_missing=True),
        "/output": modal.Volume.from_name("whisper-output", create_if_missing=True),
    },
)
def train_whisper():
    """Fonction d'entraînement qui s'exécute sur Modal"""
    import os
    import torch
    from datasets import load_dataset, Audio, Dataset
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

    MODEL_NAME = "openai/whisper-large-v3"
    OUTPUT_DIR = "/output/gilbert-whisper-large-v3-fr-v1"
    SAMPLING_RATE = 16000
    TRAIN_TEST_SPLIT = 0.95

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
        """Load datasets - Utilise des datasets qui fonctionnent vraiment"""
        print("Loading datasets...")
        all_datasets = []
        
        # Utiliser des datasets qui fonctionnent vraiment
        # FLEURS est un dataset stable de Google
        dataset_configs = [
            {
                "name": "google/fleurs",
                "config": "fr_fr",
                "text_column": "transcription",
                "audio_column": "audio",
                "split": "train",
            },
        ]
        
        for ds_config in dataset_configs:
            print(f"\n📦 Loading {ds_config['name']} ({ds_config['config']})...")
            
            try:
                # Charger le dataset
                print("   📡 Loading dataset...")
                dataset = load_dataset(
                    ds_config["name"],
                    ds_config["config"],
                    split=ds_config.get("split", "train"),
                    trust_remote_code=True,  # Nécessaire pour certains datasets
                )
                
                print(f"   ✓ Dataset loaded: {len(dataset)} samples")
                
                # Vérifier les colonnes
                print(f"   📋 Available columns: {dataset.column_names}")
                
                text_col = ds_config["text_column"]
                audio_col = ds_config["audio_column"]
                
                if text_col not in dataset.column_names:
                    print(f"   ⚠️  Text column '{text_col}' not found!")
                    # Essayer de trouver une colonne texte
                    possible_text_cols = ["transcription", "text", "sentence", "raw_text", "normalized_text"]
                    for col in possible_text_cols:
                        if col in dataset.column_names:
                            text_col = col
                            print(f"   ✓ Using text column: {text_col}")
                            break
                    else:
                        print(f"   ❌ No text column found, skipping...")
                        continue
                
                if audio_col not in dataset.column_names:
                    print(f"   ⚠️  Audio column '{audio_col}' not found!")
                    # Essayer de trouver une colonne audio
                    if "audio" in dataset.column_names:
                        audio_col = "audio"
                    elif "path" in dataset.column_names:
                        # Certains datasets ont juste le path
                        print(f"   ⚠️  Dataset has 'path' but not 'audio', trying to load...")
                        # On devra charger l'audio manuellement
                        continue
                    else:
                        print(f"   ❌ No audio column found, skipping...")
                        continue
                
                # Cast audio
                print("   🎵 Casting audio column...")
                try:
                    dataset = dataset.cast_column(
                        audio_col,
                        Audio(sampling_rate=SAMPLING_RATE)
                    )
                except Exception as e:
                    print(f"   ⚠️  Could not cast audio: {e}")
                    continue
                
                # Select and rename columns
                dataset = dataset.select_columns([audio_col, text_col])
                dataset = dataset.rename_columns({
                    audio_col: "audio",
                    text_col: "text"
                })
                
                # Filter empty texts
                dataset = dataset.filter(
                    lambda x: x["text"] is not None and len(str(x["text"]).strip()) > 0
                )
                
                print(f"   ✓ Final dataset size: {len(dataset)} samples")
                all_datasets.append(dataset)
                break  # Si on a réussi, on s'arrête
                
            except Exception as e:
                print(f"   ❌ Error loading {ds_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_datasets:
            raise ValueError("No datasets were successfully loaded!")
        
        print(f"\n📊 Using {len(all_datasets)} dataset(s)...")
        combined_dataset = all_datasets[0]
        
        print(f"   ✓ Total samples: {len(combined_dataset)}")
        
        combined_dataset = combined_dataset.shuffle(seed=42)
        split_dataset = combined_dataset.train_test_split(
            test_size=1 - TRAIN_TEST_SPLIT,
            seed=42
        )
        
        print(f"   ✓ Train samples: {len(split_dataset['train'])}")
        print(f"   ✓ Test samples: {len(split_dataset['test'])}")
        
        return split_dataset

    def prepare_dataset(example, processor):
        """Prepare a single example"""
        audio = example["audio"]
        
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        labels = processor.tokenizer(example["text"]).input_ids
        
        if isinstance(labels[0], list):
            labels = labels[0]
        
        labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "input_features": input_features.numpy(),
            "labels": labels
        }

    def compute_metrics(pred, processor, metric):
        """Compute WER metric"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        normalizer = BasicTextNormalizer()
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]
        
        wer = metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}

    # Main training
    print("🚀 Starting Whisper Large V3 Fine-tuning for French")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="/model_cache")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    print("   ✓ Model loaded")
    
    datasets = load_and_prepare_datasets()
    
    print("\n🔧 Preparing datasets...")
    
    def prepare_fn(example):
        return prepare_dataset(example, processor)
    
    train_dataset = datasets["train"].map(
        prepare_fn,
        remove_columns=datasets["train"].column_names,
        num_proc=4,
    )
    
    test_dataset = datasets["test"].map(
        prepare_fn,
        remove_columns=datasets["test"].column_names,
        num_proc=4,
    )
    
    print("   ✓ Datasets prepared")
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        push_to_hub=False,
    )
    
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, wer_metric)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )
    
    print("\n🏋️  Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training completed!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Final WER: {eval_results.get('eval_wer', 'N/A')}")
    
    return {"wer": eval_results.get('eval_wer', 'N/A'), "output_dir": OUTPUT_DIR}


@app.local_entrypoint()
def main():
    """Point d'entrée local"""
    print("🚀 Lancement de l'entraînement Whisper sur Modal...")
    result = train_whisper.remote()
    print(f"\n✅ Entraînement terminé !")
    print(f"📊 WER final: {result['wer']}")
    print(f"📁 Modèle sauvegardé dans: {result['output_dir']}")

