# Model Fine-Tuning on CPU

This project demonstrates how to fine-tune a pre-trained transformer model on a CPU using the Hugging Face `transformers`, `peft`, and `datasets` libraries, along with PyTorch. It leverages Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) to optimize the fine-tuning process for resource-constrained environments.

## Prerequisites

- Python 3.8 or higher
- A CPU-based system (no GPU required)
- Git installed for version control

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/choosechart/fine-tune-cpu.git
   cd fine-tune-cpu
   ```

2. Install dependencies using pip:
   ```bash
   pip install transformers peft datasets torch
   ```

Alternatively, use `uv` to manage dependencies:
   ```bash
   uv sync
   ```

## Project Structure

- `fine_tune.py`: Main script containing the fine-tuning logic.
- `README.md`: This file, providing project overview and instructions.
- `.gitignore`: Excludes unnecessary files (e.g., virtual environments, model checkpoints).

## Dependencies

The project relies on the following Python libraries:

- **`transformers`**: Provides pre-trained models and tokenizers for natural language processing tasks.
  - `AutoModelForSequenceClassification`: Loads a pre-trained model for sequence classification tasks.
  - `AutoTokenizer`: Handles tokenization of input text.
  - `Trainer`: Manages the training loop for fine-tuning.
  - `TrainingArguments`: Configures training hyperparameters.
- **`peft`**: Enables parameter-efficient fine-tuning.
  - `LoraConfig`: Configures LoRA for efficient model adaptation.
  - `get_peft_model`: Applies LoRA to the pre-trained model.
- **`datasets`**: Manages datasets for training and evaluation.
  - `Dataset`: Represents a single dataset.
  - `DatasetDict`: Organizes multiple datasets (e.g., train, validation, test).
- **`torch`**: PyTorch library for tensor operations and model training on CPU.

## Usage

1. Prepare your dataset in a format compatible with the `datasets` library (e.g., a `DatasetDict` with train and validation splits).
2. Configure the model and LoRA settings in `main.py`.
3. Run the fine-tuning script:
   ```bash
   uv run python fine_tune.py --output_dir "./fine_tuned_distilbert" --model_name "distilbert/distilbert-base-uncased" --data_file_path "data_dict_example.json"
   ```
   Or, if using `uv`:
   ```bash
   uv run python fine_tune.py --output_dir "./fine_tuned_distilbert" --model_name "distilbert/distilbert-base-uncased" --data_file_path "data_dict_example.json"
   ```



## Notes

- This project is optimized for CPU usage, making it suitable for environments without GPU support.
- LoRA reduces memory and computational requirements, enabling efficient fine-tuning on modest hardware.
- Ensure sufficient memory (at least 16GB RAM recommended) for handling large transformer models.

## License

This project is licensed under the MIT License.