import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from torch.optim import AdamW
from app.data.preprocessor_utils import TranscriptDataset


class FinancialHierarchicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, chunked_input, attention_masks):
        outputs = self.encoder(input_ids=chunked_input, attention_mask=attention_masks)
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]
        doc_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
        return self.regressor(doc_embedding)


def train_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    data = pd.read_pickle("motley_fool_masked.pkl")

    full_dataset = TranscriptDataset(data, tokenizer)

    # 50% Train, 40% Val, 10% Test - Going to do a lot of val
    total_size = len(full_dataset)
    train_size = int(0.5 * total_size)
    val_size = int(0.4 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FinancialHierarchicalModel().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )

    early_stop_count = 0
    min_val_loss = float("inf")

    epochs = 20
    print("Starting training...")



    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:

            input_ids = batch['input_ids'].squeeze(0).to(device)
            attention_mask = batch['attention_mask'].squeeze(0).to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_mask)  # Output [1, 1]
            loss = criterion(output.squeeze(), label)  # Squeeze to match scalar

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        predictions_log = []  # Store (error, id, pred, actual)

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].squeeze(0).to(device)
                attention_mask = batch['attention_mask'].squeeze(0).to(device)
                label = batch['label'].to(device)
                doc_id = batch['id'][0]
                text_snippet = batch['raw_text'][0]

                output = model(input_ids, attention_mask)
                loss = criterion(output.squeeze(), label)
                val_loss += loss.item()

                # Calculate absolute error for "Worst Transcripts" logic
                pred_val = output.item()
                actual_val = label.item()
                error = abs(pred_val - actual_val)

                predictions_log.append({
                    'id': doc_id,
                    'text': text_snippet,
                    'prediction': pred_val,
                    'actual': actual_val,
                    'error': error
                })

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Update Scheduler
        scheduler.step(avg_val_loss)


        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), 'transcript_model.pth')
        else:
            early_stop_count += 1
        if early_stop_count > 5:
            print("Early stopping")
            break



    # --- End of Training: Print 10 Worst Transcripts (from Validation set) ---
    print("\n" + "=" * 30)
    print("ANALYSIS: Top 10 Worst Predictions (Validation Set)")
    print("=" * 30)

    sorted_preds = sorted(predictions_log, key=lambda x: x['error'], reverse=True)

    for i, item in enumerate(sorted_preds[:10]):
        print(f"Rank {i + 1}: {item['id']}")
        print(f"  Error: {item['error']:.4f}")
        print(f"  Predicted: {item['prediction']:.4f} | Actual: {item['actual']:.4f}")
        print(f"  Snippet: {item['text']}")
        print("-" * 20)
