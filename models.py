import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

loss_fn = nn.CrossEntropyLoss()

class MLUBModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels, inference=False, linear_dropout=0.1, base_model_dropout=None):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels

        if base_model_dropout:
            print(f"[important] changing `base_model_dropout` to {base_model_dropout}")
            config.hidden_dropout_prob = base_model_dropout
            config.attention_probs_dropout_prob = base_model_dropout

        if inference:
            self.base_model = AutoModel.from_config(config=config)
        else:
            self.base_model = AutoModel.from_pretrained(model_name_or_path, config=config)

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(linear_dropout),
            nn.Linear(config.hidden_size, num_labels),
        )
        
    def forward(self, input_ids, attention_mask, start_end_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask)
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

class MLUBModelWithMeaningAttention(nn.Module):
    def __init__(self, model_name_or_path, num_labels, meaning_input_ids, meaning_attention_mask, inference=False, linear_dropout=0.1, base_model_dropout=None):
        super().__init__()

        # assuming those tensors have already got device configured
        self.meaning_input_ids = meaning_input_ids
        self.meaning_attention_mask = meaning_attention_mask

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels

        if base_model_dropout:
            print(f"[important] changing `base_model_dropout` to {base_model_dropout}")
            config.hidden_dropout_prob = base_model_dropout
            config.attention_probs_dropout_prob = base_model_dropout

        if inference:
            self.base_model = AutoModel.from_config(config=config)
        else:
            self.base_model = AutoModel.from_pretrained(model_name_or_path, config=config)

        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(linear_dropout),
            # nn.Linear(config.hidden_size, num_labels),
        )
        
        self.meaning_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(linear_dropout),
            # nn.Linear(config.hidden_size, num_labels),
        )
        
    def forward(self, input_ids, attention_mask, start_end_mask, labels=None):
        # input text
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        x = outputs.last_hidden_state
        b, s, h = x.size()
        start_end_mask = start_end_mask.unsqueeze(-1).expand(b, s, h)
        x = x * start_end_mask
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        
        # meaning representation
        meaning_outputs = self.base_model(
            input_ids=self.meaning_input_ids, 
            attention_mask=self.meaning_attention_mask)
        meaning_x = meaning_outputs.pooler_output
        meaning_x = self.meaning_linear(meaning_x)

        # attention-like
        logits = torch.matmul(x, meaning_x.T)

        # loss
        loss = loss_fn(logits, labels) if labels is not None else None
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    model_name_or_path = "bayartsogt/mongolian-roberta-base"
    num_labels = 13 # this is 69 in our case
    batch_size = 4; max_len = 20
    vocab_size = 32000

    print("------ MLUBModel ------")
    model  =MLUBModel(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        inference=True, 
        linear_dropout=0.1, 
        base_model_dropout=None
    )

    # Input: B, S
    # Output:
    #   - loss: 0
    #   - logits: B, L
    input_ids = torch.randint(0, num_labels, (batch_size, max_len))
    attention_mask = torch.randint(0, 1, (batch_size, max_len))
    start_end_mask = torch.randint(0, 1, (batch_size, max_len))
    
    output = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        start_end_mask = start_end_mask,
    )
    
    print("input_ids.size():", input_ids.size())
    print("attention_mask.size():", attention_mask.size())
    print("start_end_mask.size():", start_end_mask.size())
    print("------")
    print("output.logits.size():", output.logits.size())
    print("output.loss:", output.loss)
    print("-----------------------")
    

    """
    MLUBModelWithMeaningAttention
        forward:
            input_ids: b, s
            attention_mask: b, s
            start_end_mask: b, s
            meaning_input_ids: l, s
            meaning_attention_mask: l, s
    """
    print("------ MLUBModelWithMeaningAttention ------")
    
    input_ids = torch.randint(0, vocab_size, (batch_size, max_len))
    attention_mask = torch.randint(0, 1, (batch_size, max_len))
    start_end_mask = torch.randint(0, 1, (batch_size, max_len))
    meaning_input_ids = torch.randint(0, vocab_size, (num_labels, max_len))
    meaning_attention_mask = torch.randint(0, 1, (num_labels, max_len))

    print("             input_ids:", input_ids.size())
    print("        attention_mask:", attention_mask.size())
    print("        start_end_mask:", start_end_mask.size())
    print("     meaning_input_ids:", meaning_input_ids.size())
    print("meaning_attention_mask:", meaning_attention_mask.size())
    print("------")

    model = MLUBModelWithMeaningAttention(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        inference=True, 
        linear_dropout=0.1, 
        base_model_dropout=None,
        meaning_input_ids = meaning_input_ids,
        meaning_attention_mask = meaning_attention_mask,
    ) 

    output = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        start_end_mask = start_end_mask,
    )

    print("output.logits.size():", output.logits.size())
    print("         output.loss:", output.loss)
    print("------")