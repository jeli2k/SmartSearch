import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, AdamW
from transformers import get_scheduler
from tqdm import tqdm

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


def main(data_dir, model_output_dir):
    """Fine-tune the model using the provided data directory."""
    # Load the training data
    train_data_path = os.path.join(data_dir, 'train_data.csv')
    train_data = load_data(train_data_path)

    # Check if train_data is empty
    if not train_data:
        print("Error: No valid rows found after cleaning.")
    else:
        print(f"Validated dataset with {len(train_data)} rows.")

    # Initialize the reader model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForQuestionAnswering.from_pretrained("microsoft/codebert-base")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # Move model to the selected device
    
    # Create Dataset
    dataset = CustomDataset(train_data, tokenizer)
    print(f"Dataset size: {len(dataset)}")  # Debug statement

    # Ensure there are enough samples to split
    if len(dataset) < 2:
        print("Warning: Not enough samples to create both training and validation sets. Exiting...")
        return

    # Proceed with dataset splitting
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch size
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Adjust batch size
    # Set up the optimizer
    # Adjust learning rate
    optimizer = AdamW([
    {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': 1e-5},
    {'params': model.roberta.encoder.layer[6:].parameters(), 'lr': 3e-5},
    {'params': model.qa_outputs.parameters(), 'lr': 3e-5}
])
    
    # Fine-tuning loop
    num_epochs = 6  # Adjust Epochs
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(
        "linear",  # Linear decay with warmup
        optimizer=optimizer,
        num_warmup_steps=0,  # Adjust warmup steps as needed
        num_training_steps=total_steps
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15, verbose=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()  # Clear previous gradients
            
            # Move inputs to the selected device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            # **Gradient clipping**
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

        # Average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                
                # Calculate loss
                total_val_loss += outputs.loss.item()

        # Average validation loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Check early stopping condition
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            break

        # Save the model after each epoch
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for codebase search")
    parser.add_argument("--data_dir", type=str, help="Directory containing training data")
    parser.add_argument("--model_output_dir", type=str, default="train/trained_models/", help="Directory to save the fine-tuned model")

    args = parser.parse_args()
    main(data_dir='train', model_output_dir='train/trained_models/codebert_finetuned')
