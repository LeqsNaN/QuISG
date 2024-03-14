import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizerFast
from utils.config import *


class KeyModel(nn.Module):
    def __init__(self, config):
        super(KeyModel, self).__init__()
        self.model = ElectraModel.from_pretrained(args.model_name)
        self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
        self.classifier = nn.Linear(2*config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                utterance_range=None,
                question_mask=None,
                label=None,
                label_mask=None):
        # question_range: [batch_size, 512]
        # utterance_range: [batch_size, num_tri, 512]
        # label, label_mask: [batch_size, num_tri]
        transformer_outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
        # [batch_size, 512, hidden_size]
        hidden_states = transformer_outputs[0]

        question_reps = hidden_states * question_mask.unsqueeze(2)
        # [batch_size, hidden_size]
        question_reps = question_reps.sum(1) / (torch.sum(question_mask, dim=1, keepdim=True) + 1e-20)

        # [batch_size, num_tri, 512, hidden_size]
        tri_utter_reps = hidden_states.unsqueeze(1) * utterance_range.unsqueeze(3)
        # [batch_size, num_tri, hidden_size]
        tri_utter_reps = tri_utter_reps.sum(2) / (torch.sum(utterance_range, dim=2, keepdim=True) + 1e-20)
        bsz, num_tri, hsz = tri_utter_reps.shape

        # [batch_size, num_tri, 2*hidden_size]
        pair_reps = torch.cat([question_reps.unsqueeze(1).expand(bsz, num_tri, hsz), tri_utter_reps], dim=2)

        logits = self.sigmoid(self.classifier(pair_reps))

        loss_fn = nn.BCELoss(reduction='none')

        loss = loss_fn(logits.view(-1), label.view(-1))
        loss = (loss * label_mask.view(-1)).sum() / label_mask.sum()

        return loss, logits
