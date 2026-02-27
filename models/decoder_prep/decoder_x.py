import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class DecoderX(nn.Module):
    def __init__(self, model_name = "t5-small"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, encoder_hidden_states, attention_mask, labels):
        outputs = self.model(
            encoder_outputs = (encoder_hidden_states, ),
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True
        )

        return outputs.loss, outputs.logits