import torch
import torch.nn as nn
import torch.nn.functional as F

class ForgettingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        slot_pooling: nn.Module,
        u_head: nn.Module,
        v_head: nn.Module,
        decoder_x: nn.Module,
        decoder_y: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.slot_pooling = slot_pooling
        self.u_head = u_head
        self.v_head = v_head
        self.decoder_x = decoder_x
        self.decoder_y = decoder_y


    def info_nce_loss(self, u, upos, temperature=0.1):
        """
        InfoNCE loss. TODO: look for other loss functions
        """
        u = F.normalize(u, dim=-1)
        upos = F.normalize(upos, dim=-1)
        logits = torch.matmul(u, upos.T) / temperature
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    

    def forward(self, batch):
        device = next(self.parameters()).device
        lambda_u, lambda_x, lambda_y = 1.0, 1.0, 1.0

        input_ids = batch["x_input_ids"].to(device)
        attention_mask = batch["x_attention"].to(device)
        xpos_input_ids = batch["xpos_input_ids"].to(device)
        xpos_attention_mask = batch["xpos_attention"].to(device)
        labels_x = batch["x_input_ids"].to(device)
        labels_y = batch["y_input_ids"].to(device)
        y_attention_mask = batch["y_attention"].to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        Hpos = self.encoder(input_ids=xpos_input_ids, attention_mask=xpos_attention_mask)

        outputs = self.slot_pooling(H, attention_mask)
        pos_outputs = self.slot_pooling(Hpos, xpos_attention_mask)

        u = self.u_head(outputs)
        upos = self.u_head(pos_outputs)
        v0 = self.v_head(outputs)

        B, L, _ = v0.shape
        slot_mask = torch.ones((B,L), device = device)

        loss_x, logits_x = self.decoder_x(v0, slot_mask, labels_x)
        loss_y, logits_y = self.decoder_y(u, labels_y)
        loss_nce = self.info_nce_loss(u, upos)

        total_loss = (
            lambda_u * loss_nce +
            lambda_x * loss_x +
            lambda_y * loss_y
        )

        return total_loss, logits_x, logits_y