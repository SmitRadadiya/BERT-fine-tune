import torch
import torch.nn as nn

class BERT(nn.Module):

    def __init__(self, bert, h_dim, op_dim) -> None:
        super(BERT, self).__init__()

        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Linear(786, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, op_dim)
        )

        
    def forward(self, ids, mask):
        num_heads = self.bert.config.num_attention_heads
        head_mask = [1] * num_heads
        last_hidden_state, _  = self.bert(input_ids = ids, attention_mask = mask, head_mask=torch.tensor([head_mask]))
        cls_token = last_hidden_state[:, 0, :]
        output = self.classifier(cls_token)
        return output
        

