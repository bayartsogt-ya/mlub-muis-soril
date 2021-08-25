import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

loss_fn = nn.CrossEntropyLoss()

class MLUBModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels, inference=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels

        if inference:
            self.base_model = AutoModel.from_config(config=config)
        else:
            self.base_model = AutoModel.from_pretrained(model_name_or_path, config=config)

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, num_labels),
        )
        
    def forward(self, input_ids, attention_mask, start_end_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask)
        x = outputs.last_hidden_state
        b, s, h = x.size()
        start_end_mask = start_end_mask.unsqueeze(-1).expand(b, s, h)
        x = x * start_end_mask
        
        x = torch.sum(x, dim=1)
        logits = self.linear(x)
        loss = loss_fn(logits, labels) if labels is not None else None
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
