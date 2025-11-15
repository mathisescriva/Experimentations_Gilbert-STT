#!/usr/bin/env python3
"""
Version de TEST du script de fine-tuning - limite les données pour validation rapide
Utilisez ce script pour tester le pipeline avant de lancer l'entraînement complet
"""

import os
import torch
from datasets import load_dataset, DatasetDict, Audio
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
from tqdm import tqdm


# Configuration TEST - LIMITE LES DONNÉES
MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "./gilbert-whisper-large-v3-fr-v1-test"
SAMPLING_RATE = 16000
TRAIN_TEST_SPLIT = 0.95
MAX_SAMPLES_PER_DATASET = 100  # LIMITE POUR TEST

# Datasets to use
DATASETS_CONFIG = [
    {
        "name": "fsicoli/common_voice_17_0",
        "config": "fr",
        "text_column": "sentence",
        "audio_column": "audio",
    },
    {
        "name": "facebook/voxpopuli",
        "config": "fr",
        "text_column": None,
        "audio_column": "audio",
    },
]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper fine-tuning with proper padding"""
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels and replace padding token id's with -100
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Keep -100 for tokens to ignore in loss computation
        batch["labels"] = labels

        return batch


def detect_text_column(dataset, possible_columns=["normalized_text", "raw_text", "text", "sentence"]):
    """Detect the correct text column in the dataset"""
    for col in possible_columns:
        if col in dataset.column_names:
            return col
    raise ValueError(f"Could not find text column in {possible_columns}. Available columns: {dataset.column_names}")


def load_and_prepare_datasets():
    """Load and prepare all datasets - VERSION TEST avec limite"""
    print("Loading datasets (TEST MODE - limited samples)...")
    all_datasets = []
    
    for ds_config in DATASETS_CONFIG:
        print(f"\n📦 Loading {ds_config['name']} ({ds_config['config']})...")
        
        try:
            dataset = load_dataset(
                ds_config["name"],
                ds_config["config"],
                split="train",
                streaming=False,
            )
            
            # LIMITE POUR TEST
            if len(dataset) > MAX_SAMPLES_PER_DATASET:
                print(f"   ⚠️  Limiting to {MAX_SAMPLES_PER_DATASET} samples for test")
                dataset = dataset.select(range(MAX_SAMPLES_PER_DATASET))
            
            # Detect text column if not specified
            text_col = ds_config["text_column"]
            if text_col is None:
                text_col = detect_text_column(dataset)
                print(f"   ✓ Detected text column: {text_col}")
            
            # Verify columns exist
            if text_col not in dataset.column_names:
                print(f"   ⚠️  Warning: {text_col} not found, skipping...")
                continue
            
            if ds_config["audio_column"] not in dataset.column_names:
                print(f"   ⚠️  Warning: {ds_config['audio_column']} not found, skipping...")
                continue
            
            # Cast audio to Audio feature with correct sampling rate
            dataset = dataset.cast_column(
                ds_config["audio_column"],
                Audio(sampling_rate=SAMPLING_RATE)
            )
            
            # Select only needed columns and rename for consistency
            dataset = dataset.select_columns([ds_config["audio_column"], text_col])
            dataset = dataset.rename_columns({
                ds_config["audio_column"]: "audio",
                text_col: "text"
            })
            
            # Filter out None or empty texts
            dataset = dataset.filter(
                lambda x: x["text"] is not None and len(x["text"].strip()) > 0
            )
            
            print(f"   ✓ Loaded {len(dataset)} samples")
            all_datasets.append(dataset)
            
        except Exception as e:
            print(f"   ❌ Error loading {ds_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded!")
    
    # Concatenate all datasets
    print(f"\n📊 Concatenating {len(all_datasets)} datasets...")
    combined_dataset = all_datasets[0]
    for ds in all_datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)
    
    print(f"   ✓ Total samples: {len(combined_dataset)}")
    
    # Shuffle and split
    combined_dataset = combined_dataset.shuffle(seed=42)
    split_dataset = combined_dataset.train_test_split(
        test_size=1 - TRAIN_TEST_SPLIT,
        seed=42
    )
    
    print(f"   ✓ Train samples: {len(split_dataset['train'])}")
    print(f"   ✓ Test samples: {len(split_dataset['test'])}")
    
    return split_dataset


def prepare_dataset(example, processor):
    """Prepare a single example of audio and text for training"""
    # Load audio array
    audio = example["audio"]
    
    # Compute input features
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    
    # Tokenize text to create labels
    labels = processor.tokenizer(example["text"]).input_ids
    
    # Flatten labels list (tokenizer returns list of lists)
    if isinstance(labels[0], list):
        labels = labels[0]
    
    # Replace padding token id with -100 (ignored in loss)
    labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
    
    return {
        "input_features": input_features.numpy(),
        "labels": labels
    }


def compute_metrics(pred, processor, metric):
    """Compute WER metric"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad_token_id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize text
    normalizer = BasicTextNormalizer()
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]
    
    # Compute WER
    wer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


def main():
    """Main training function - VERSION TEST"""
    print("🚀 Starting Whisper Large V3 Fine-tuning for French (TEST MODE)")
    print("=" * 60)
    print("⚠️  TEST MODE: Limited samples and epochs for quick validation")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cpu":
        print("⚠️  Warning: Training on CPU will be very slow. Consider using GPU cloud.")
    
    # Load processor and model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Configure decoder (as specified in requirements)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    print("   ✓ Model loaded")
    
    # Load and prepare datasets
    datasets = load_and_prepare_datasets()
    
    # Prepare datasets
    print("\n🔧 Preparing datasets...")
    
    def prepare_fn(example):
        return prepare_dataset(example, processor)
    
    train_dataset = datasets["train"].map(
        prepare_fn,
        remove_columns=datasets["train"].column_names,
        num_proc=2,  # Reduced for test
    )
    
    test_dataset = datasets["test"].map(
        prepare_fn,
        remove_columns=datasets["test"].column_names,
        num_proc=2,  # Reduced for test
    )
    
    print("   ✓ Datasets prepared")
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Training arguments - MODIFIÉS POUR TEST
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # Reduced for test
        per_device_eval_batch_size=1,  # Reduced for test
        gradient_accumulation_steps=2,  # Reduced for test
        learning_rate=1e-5,
        num_train_epochs=0.1,  # Just a few steps for test
        max_steps=5,  # Limit to 5 steps for quick test
        fp16=False,  # Disable for CPU compatibility
        evaluation_strategy="steps",
        eval_steps=3,
        save_strategy="steps",
        save_steps=3,
        logging_steps=1,
        report_to="none",  # Disable tensorboard for test
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=1,
        push_to_hub=False,
    )
    
    # Create compute_metrics function with processor and metric
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, wer_metric)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )
    
    # Train
    print("\n🏋️  Starting training (TEST - limited steps)...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training completed (TEST MODE)!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("\n💡 If this worked, you can now run the full training on GPU cloud")
    
    # Final evaluation
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Final WER: {eval_results.get('eval_wer', 'N/A')}")


if __name__ == "__main__":
    main()

