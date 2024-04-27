import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class RubertTinyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to(device)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_trained_output_len = self.bert.encoder.layer[1].output.dense.out_features
        self.linear = torch.nn.Linear(self.pre_trained_output_len, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.linear(self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output)