import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import T5Tokenizer
import json

from data_loader.data_loading import MSRDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.decoder_prep.decoder_y import DecoderY
from models.forgetting_model import ForgettingModel


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss, _, _ = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    sample_outputs = None
    
    for i, batch in enumerate(dataloader):
        loss, logits_x, logits_y = model(batch)
        total_loss += loss.item()
        if i == 0:
            sample_outputs = (batch, logits_x, logits_y)
            
    return total_loss / len(dataloader), sample_outputs


def log_sample_outputs(sample_outputs, tokenizer, epoch):
    batch, logits_x, logits_y = sample_outputs

    pred_ids_x = torch.argmax(logits_x, dim=-1)
    pred_ids_y = torch.argmax(logits_y, dim=-1)
    decoded_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
    decoded_y = tokenizer.batch_decode(pred_ids_y, skip_special_tokens=True)

    original_x = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
    original_y = tokenizer.batch_decode(batch["y_input_ids"], skip_special_tokens=True)

    results = []
    for i in range(len(decoded_x)):
        results.append({
            "original_x": original_x[i],
            "v0": decoded_x[i],
            "original_y": original_y[i],
            "u": decoded_y[i]
        })

    with open(f"./final_outputs/epoch_{epoch+1}_samples.json", "w") as f:
        json.dump(results, f, indent=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    train_dataset = MSRDataset("./dataset_prep/outputs/first_100_train.json")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRDataset("./dataset_prep/outputs/first_100_test.json")
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim = encoder.hidden_dim_size, num_slots = 8)
    u_head = UHead(hidden_dim = encoder.hidden_dim_size, output_dim = 128)
    v_head = VHead(hidden_dim = encoder.hidden_dim_size)
    decoder_x = DecoderX()
    decoder_y = DecoderY(hidden_dim = encoder.hidden_dim_size, u_dim = 128, num_slots = 8)

    model = ForgettingModel(
        encoder = encoder,
        slot_pooling = slot_pool,
        u_head = u_head,
        v_head = v_head,
        decoder_x = decoder_x,
        decoder_y = decoder_y,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    epochs = 500

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, sample_outputs = validate_epoch(model, val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("-" * 30)

        if sample_outputs:
            log_sample_outputs(sample_outputs, tokenizer, epoch)


if __name__ == "__main__":
    main()
