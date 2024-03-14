from simcse import SimCSE
import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# smodel = SimCSE('/home/ljn/pretrained/sup-simcse-roberta-large')


class SModel(nn.Module):
    def __init__(self):
        super(SModel, self).__init__()
        self.simcse = SimCSE('/mnt/sdb/ljn/pretrained/sup-simcse-roberta-large')
        self.tokenizer = self.simcse.tokenizer
        self.model = self.simcse.model
        self.classifier = nn.Linear(2*1024, 1)
        self.register_buffer('weight', torch.tensor([0.10208, 1.89792]))

    def forward(self, utterance_input_ids, utterance_attention_mask,
                question_input_ids, question_attention_mask, label):
        utterance_output = self.model(utterance_input_ids, utterance_attention_mask).pooler_output
        question_output = self.model(question_input_ids, question_attention_mask).pooler_output
        logits = torch.sigmoid(self.classifier(torch.cat([utterance_output, question_output], dim=1))).squeeze(1)
        loss_fn = nn.BCELoss(reduction='none')
        loss = loss_fn(logits, label)
        label_idx = label.data.long()
        weight_gather = torch.gather(self.weight, 0, label_idx).to(loss.device)
        loss = (loss * weight_gather).mean()
        return logits, loss


class SDataset(Dataset):
    def __init__(self, data_path, process_path, tokenizer):
        super(SDataset, self).__init__()
        if os.path.exists(process_path):
            data = pickle.load(open(process_path, 'rb'), encoding='utf-8')
            self.q_u_ids = data[0]
            self.q_list = data[1]
            self.u_list = data[2]
            self.question_input_ids = data[3]
            self.question_attention_mask = data[4]
            self.utterance_input_ids = data[5]
            self.utterance_attention_mask = data[6]
            self.label = data[7]
        else:
            question_conversation = json.load(open(data_path, 'r', encoding='utf-8'))
            self.q_u_ids = []
            self.q_list = []
            self.u_list = []
            self.question_input_ids = []
            self.question_attention_mask = []
            self.utterance_input_ids = []
            self.utterance_attention_mask = []
            self.label = []
            for key, value in question_conversation.items():
                question = value['question']
                utterances = value['utterances']
                target = value['target_uids']
                for k, v in utterances.items():
                    ki = int(k)
                    utterance_id = key + '-' + k
                    utterance = v['utterance_with_speaker']
                    self.q_u_ids.append(utterance_id)
                    self.q_list.append(question)
                    self.u_list.append(utterance)
                    question_encoded = tokenizer(question, max_length=32, truncation=True)
                    self.question_input_ids.append(question_encoded.input_ids)
                    self.question_attention_mask.append(question_encoded.attention_mask)
                    utterance_encoded = tokenizer(utterance, max_length=50, truncation=True)
                    self.utterance_input_ids.append(utterance_encoded.input_ids)
                    self.utterance_attention_mask.append(utterance_encoded.attention_mask)
                    if ki in target:
                        self.label.append(1)
                    else:
                        self.label.append(0)
            pickle.dump([self.q_u_ids, self.q_list, self.u_list, self.question_input_ids, self.question_attention_mask,
                         self.utterance_input_ids, self.utterance_attention_mask, self.label], open(process_path, 'wb'))

    def __getitem__(self, item):
        uid = self.q_u_ids[item]
        question = self.q_list[item]
        utterance = self.u_list[item]
        q_input_id = torch.tensor(self.question_input_ids[item], dtype=torch.long)
        q_attention_msk = torch.tensor(self.question_attention_mask[item])
        u_input_id = torch.tensor(self.utterance_input_ids[item], dtype=torch.long)
        u_attention_msk = torch.tensor(self.utterance_attention_mask[item])
        lbl = torch.tensor(self.label[item], dtype=torch.float)

        return uid, question, utterance, q_input_id, q_attention_msk, u_input_id, u_attention_msk, lbl

    def __len__(self):
        return len(self.label)


def collate_fn(data):
    new_uid = []
    new_question = []
    new_utterance = []
    new_q_input_ids = []
    new_q_attention_mask = []
    new_u_input_ids = []
    new_u_attention_mask = []
    new_label = []
    for d in data:
        new_uid.append(d[0])
        new_question.append(d[1])
        new_utterance.append(d[2])
        new_q_input_ids.append(d[3])
        new_q_attention_mask.append(d[4])
        new_u_input_ids.append(d[5])
        new_u_attention_mask.append(d[6])
        new_label.append(d[7])
    new_data = {}
    new_data['uid'] = new_uid
    new_data['question'] = new_question
    new_data['utterance'] = new_utterance
    new_data['question_input_ids'] = pad_sequence(new_q_input_ids, batch_first=True, padding_value=1).cuda()
    new_data['question_attention_mask'] = pad_sequence(new_q_attention_mask, batch_first=True, padding_value=0).cuda()
    new_data['utterance_input_ids'] = pad_sequence(new_u_input_ids, batch_first=True, padding_value=1).cuda()
    new_data['utterance_attention_mask'] = pad_sequence(new_u_attention_mask, batch_first=True, padding_value=0).cuda()
    new_data['label'] = torch.stack(new_label, dim=0).cuda()

    return new_data


def get_dataloader(train_path, dev_path, test_path, tokenizer, batch_size, shuffle):
    train_process_path = 'similarity/train_processed.pkl'
    dev_process_path = 'similarity/dev_processed.pkl'
    test_process_path = 'similarity/test_processed.pkl'
    train_set = SDataset(train_path, train_process_path, tokenizer)
    dev_set = SDataset(dev_path, dev_process_path, tokenizer)
    test_set = SDataset(test_path, test_process_path, tokenizer)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader
