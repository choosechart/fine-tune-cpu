from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
import torch
import argparse
import os
import json


def main(output_dir, model_name, data_file_path):    
    if os.path.exists(data_file_path) is False:
        raise Exception(f"{data_file_path} cannot be found")
    
    with open(data_file_path) as f:
        data_dict = json.loads(f.read())   
        
    # Convert to Hugging Face Dataset with "train" split
    dataset = Dataset.from_list(data_dict)
    dataset_dict = DatasetDict({"train": dataset})

    # Optional: Create a validation split (20% of data)
    dataset_dict = dataset_dict["train"].train_test_split(test_size=0.2)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification (positive/negative)
        torch_dtype=torch.float16,  # Use FP16 for low memory
        low_cpu_mem_usage=True,
        device_map="cpu"  # Ensure CPU usage
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess function: Ensure input_ids, attention_mask, and labels

    def preprocess_function(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Short length for memory efficiency
            return_tensors=None  # Return Python lists
        )
        encodings["labels"] = examples["label"]  # Add labels from dataset
        return encodings


    # Apply preprocessing
    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

    # Remove unused columns to avoid mismatches
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention modules
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"  # Sequence classification
    )
    model = get_peft_model(model, lora_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,  # Small batch for 8GB RAM
        gradient_accumulation_steps=4,  # Simulate larger batch
        num_train_epochs=3,
        fp16=True,  # Mixed precision
        dataloader_pin_memory=False,  # CPU optimization
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=True  # Remove unused columns
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("test")  # Use "test" as validation
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune model and save output.")
    parser.add_argument("--output_dir", required=True, help="Dir to save the output.")
    parser.add_argument("--model_name", required=True, help="Hugging face model to use.")
    parser.add_argument("--data_file_path", required=True, help="Path to the data file.")

    args = parser.parse_args()
    main(args.output_dir, args.model_name, args.data_file_path)


