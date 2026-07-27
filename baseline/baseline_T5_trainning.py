from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import numpy as np

MODEL_DIR = "" #Model path
TRAIN_PATH = "" #Training dataset
VAL_PATH = "" # Validation dataset

# token budget applies (input max=106 tok measured there; this run uses a
# subset of the same source examples, so it can only be shorter).
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 200

OUTPUT_DIR = "" # Model output path


def main():
    raw_datasets = load_dataset(
        "json",
        data_files={"train": TRAIN_PATH, "validation": VAL_PATH},
    )
    train_ds = raw_datasets["train"]
    eval_ds = raw_datasets["validation"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # See T5_training.py -- t5-base's stock vocab has no token for "{"/"}"
    num_added = tokenizer.add_tokens(["{", "}"])
    print(f"Added {num_added} new tokens to tokenizer for Cypher braces")
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def preprocess_function(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

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

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact_matches = [
            int(p.strip() == l.strip())
            for p, l in zip(decoded_preds, decoded_labels)
        ]
        return {"exact_match": sum(exact_matches) / len(exact_matches)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch size = 16
        learning_rate=2e-4,
        weight_decay=0.01,

        warmup_ratio=0.05,
        lr_scheduler_type="linear",

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # See T5_training.py -- fp16 reliably produces NaN/Inf gradients on
        # this V100 (no bf16 tensor cores; T5 overflows fp16's ~65504 max).
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print("==== Final evaluation metrics ====")
    print(eval_metrics)

    final_dir = OUTPUT_DIR + "/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
