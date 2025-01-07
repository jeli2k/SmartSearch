import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, AdamW
from transformers import get_scheduler
from tqdm import tqdm
import itertools


class CustomDataset(Dataset):
    def __init__(self, train_data, tokenizer):
        self.train_data = train_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        question = self.train_data[idx]["question"]
        answer_text = self.train_data[idx]["answer"]
        context = self.train_data[idx].get("context", "Default context for training.")

        # Tokenize the context and answer independently to find token positions
        tokenized_context = self.tokenizer(context, add_special_tokens=False)
        tokenized_answer = self.tokenizer(answer_text, add_special_tokens=False)

        # Find the start and end token positions for the answer in the tokenized context
        answer_ids = tokenized_answer['input_ids']
        context_ids = tokenized_context['input_ids']

        # Ensure the answer tokens are present in the context tokens
        try:
            start_position = context_ids.index(answer_ids[0]) if answer_ids[0] in context_ids else 0
            end_position = start_position + len(answer_ids) - 1
        except ValueError:
            start_position = 0  # Default to 0 if answer is not found in context
            end_position = 0

        # Encode both question and context for training
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_position),
            "end_positions": torch.tensor(end_position)
        }

def hyperparameter_tuning(data_dir, model_output_dir, learning_rates, batch_sizes, warmup_steps_ratios, num_epochs=3):
    """Perform grid search for hyperparameter tuning."""
    # Load training data
    train_data_path = os.path.join(data_dir, 'train_data.csv')
    train_data = load_data(train_data_path)

    if not train_data:
        print("No valid data found. Exiting...")
        return

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForQuestionAnswering.from_pretrained("microsoft/codebert-base")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare dataset
    dataset = CustomDataset(train_data, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Grid search over hyperparameters
    best_loss = float('inf')
    best_config = None
    results = []

    for lr, batch_size, warmup_ratio in itertools.product(learning_rates, batch_sizes, warmup_steps_ratios):
        print(f"\nTesting configuration: LR={lr}, Batch Size={batch_size}, Warmup Ratio={warmup_ratio}")

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # Evaluate on validation set
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)

                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions
                    )
                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Update best configuration
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_config = {'learning_rate': lr, 'batch_size': batch_size, 'warmup_ratio': warmup_ratio}
                model.save_pretrained(model_output_dir)
                tokenizer.save_pretrained(model_output_dir)

        results.append({'lr': lr, 'batch_size': batch_size, 'warmup_ratio': warmup_ratio, 'val_loss': avg_val_loss})

    # Output the best configuration
    print("\nBest Configuration:")
    print(best_config)
    print(f"Best Validation Loss: {best_loss:.4f}")

    return results

def load_data(file_path):
    """Load training data from a CSV file."""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"Loaded {len(df)} rows from the CSV.")
        train_data = []
        for i, row in df.iterrows():
            if pd.isna(row['question']) or pd.isna(row['answer']) or pd.isna(row.get('context')):
                print(f"Skipping row {i} due to NaN values: {row}")
                continue
            train_data.append({
                "question": str(row['question']),
                "answer": str(row['answer']),
                "context": str(row.get('context', "Default context for training."))
            })
        print(f"Successfully loaded {len(train_data)} training samples.")
        return train_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    
class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True
                


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for CodeBERT")
    parser.add_argument("--data_dir", type=str, default="train/", help="Directory containing training data")
    parser.add_argument("--model_output_dir", type=str, default="train/trained_models/", help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [4, 8, 16]
    warmup_steps_ratios = [0.1, 0.2]

    results = hyperparameter_tuning(
        data_dir=args.data_dir,
        model_output_dir=args.model_output_dir,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        warmup_steps_ratios=warmup_steps_ratios
    )