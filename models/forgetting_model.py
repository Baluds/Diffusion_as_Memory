import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gpsi import GPsi

class ForgettingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        slot_pooling: nn.Module,
        u_head: nn.Module,
        v_head: nn.Module,
        decoder_x: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.slot_pooling = slot_pooling
        self.u_head = u_head
        self.v_head = v_head
        self.decoder_x = decoder_x
        self.gpsi = GPsi(u_dim=128, d_model=self.encoder.hidden_dim_size)

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
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, batch):
        device = self.device
        lambda_u, lambda_x = 1.0, 1.0

        input_ids = batch["x_input_ids"].to(device)
        attention_mask = batch["x_attention"].to(device)
        xpos_input_ids = batch["xpos_input_ids"].to(device)
        xpos_attention_mask = batch["xpos_attention"].to(device)
        labels_x = batch["x_input_ids"].to(device)
        # labels_y = batch["y_input_ids"].to(device)
        # y_attention_mask = batch["y_attention"].to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        Hpos = self.encoder(input_ids=xpos_input_ids, attention_mask=xpos_attention_mask)
        # print("shape of outputs after encoder", H.shape) # b, 64, 512

        outputs = self.slot_pooling(H, attention_mask)
        # print("shape of outputs after slot pooling", outputs.shape) # b, 8, 512
        pos_outputs = self.slot_pooling(Hpos, xpos_attention_mask)

        u = self.u_head(outputs)
        upos = self.u_head(pos_outputs)
        v0 = self.v_head(outputs)

        B, L, _ = v0.shape
        slot_mask = torch.ones((B,L), device = device)

        use_u_for_v0 = True  # can make this a config later, this is an optional flag for u+v0 instead of v0

        if use_u_for_v0:
            v0 = self.gpsi(v0, u)

        loss_x, logits_x = self.decoder_x(v0, slot_mask, labels_x)
        # loss_y, logits_y = self.decoder_y(u, labels_y)
        loss_nce = self.info_nce_loss(u, upos)

        total_loss = (
            lambda_u * loss_nce +
            lambda_x * loss_x
            # lambda_y * loss_y
        )

        return total_loss, logits_x, loss_nce, loss_x

    @torch.no_grad()
    def encode_latents(self, batch):
        """
        Extract latent representations (u, v0) without running decoders.
        """
        device = self.device

        input_ids = batch["x_input_ids"].to(device)
        attention_mask = batch["x_attention"].to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.slot_pooling(H, attention_mask)

        u = self.u_head(outputs)    # [B, 128]
        v0 = self.v_head(outputs)   # [B, 8, 512]

        return u, v0