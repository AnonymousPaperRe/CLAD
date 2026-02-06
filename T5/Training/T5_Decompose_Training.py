# train_t5.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import numpy as np

MODEL_DIR = ""
DATA_PATH = ""

# Shorter lengths -> less memory, enough for your questions
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

# Save big stuff to scratch, not to home (quota!)
OUTPUT_DIR = ""


def main():
    # 1) Load dataset from JSON
    raw_datasets = load_dataset(
        "json",
        data_files={"train": DATA_PATH},
        split="train"
    )

    # Split 90% train / 10% validation
    raw_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)
    train_ds = raw_datasets["train"]
    eval_ds = raw_datasets["test"]

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # --- ADD SPECIAL TOKENS HERE ---
    # Define the node tags you plan to use in your dataset
    special_tokens_dict = {'additional_special_tokens': ['[node_1]', '[node_2]', '[node_3]', '[node_4]']}
    
    # Add to tokenizer
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to tokenizer")

    # Resize model embeddings to match the new tokenizer length
    model.resize_token_embeddings(len(tokenizer))
    # -------------------------------   

    # Save GPU memory
    model.config.use_cache = False     
    model.gradient_checkpointing_enable()

    # Optional: prefix to tell T5 what to do
    PREFIX = "decompose: "

    def preprocess_function(batch):
        inputs = [PREFIX + x for x in batch["input"]]
        targets = batch["output"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            # padding="max_length",
        )

        # Deprecated but fine for this transformers version
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length",
            )

        # Replace padding token id with -100 so they are ignored in loss
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
            for seq in labels_ids
        ]

        model_inputs["labels"] = labels_ids
        return model_inputs

    # 3) Tokenize datasets
    train_tokenized = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tokenized = eval_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # 4) Data collator (for seq2seq)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 5) Training arguments (safer on memory, no metrics)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # logging & saving (no frequent checkpoints to avoid quota issues)
        logging_steps=50,
        save_steps=1000000,        # effectively don't save mid-training
        save_total_limit=1,

        num_train_epochs=5,        # reduced from 5
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch size = 16
        learning_rate=2e-4,
        weight_decay=0.01,

        warmup_ratio=0.05,
        lr_scheduler_type="linear",

        # evaluation_strategy="epoch",  # <-- add this
        # predict_with_generate=False,  # set True later when you want text metrics

        fp16=True,                 # V100 supports fp16
        report_to="none",          # no wandb by default
    )

    # 6) Trainer (no compute_metrics for now)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 7) Train
    trainer.train()

    # 8) Final evaluation on validation set (gives eval_loss etc.)
    eval_metrics = trainer.evaluate()
    print("==== Final evaluation metrics ====")
    print(eval_metrics)

    # 9) Save final model + tokenizer to scratch
    final_dir = OUTPUT_DIR + "/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
