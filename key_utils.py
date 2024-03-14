import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
from utils.config import *
from tqdm import tqdm
import stanza


class ConvExample:
    def __init__(self, context, utterances, question, qid, key_utterance, utterances_cased=None):
        self.context = context
        self.utterances = utterances
        self.utterances_cased = utterances_cased
        self.question = question
        self.qid = qid
        self.key_utterance = key_utterance


class ConvFeature:
    def __init__(self, qid, input_ids, token_type_ids, attention_mask, p_mask,
                 context, key_utterance, question_len, utterance_range, label):
        self.qid = qid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.p_mask = p_mask
        self.context = context
        self.key_utterance = key_utterance
        self.question_len = question_len
        self.utterance_range = utterance_range
        self.label = label


class ConvDataset(Dataset):
    def __init__(self, features, max_length=512):
        super(ConvDataset, self).__init__()
        self.features = features
        self.max_length = max_length

    def __getitem__(self, item):
        data_info = dict()
        data_info['qid'] = self.features[item].qid
        data_info['input_ids'] = torch.tensor(self.features[item].input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[item].attention_mask)
        data_info['token_type_ids'] = torch.tensor(self.features[item].token_type_ids, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[item].p_mask, dtype=torch.float)
        data_info['context'] = self.features[item].context
        data_info['key_utterance'] = self.features[item].key_utterance
        question_len = self.features[item].question_len
        data_info['question_mask'] = torch.cat([torch.ones(question_len), torch.zeros(self.max_length-question_len)], dim=0)
        utterance_range = self.features[item].utterance_range
        utterance_mask = []
        for ur in utterance_range:
            utterance_mask.append(torch.cat([torch.zeros(ur[0]), torch.ones(ur[1]-ur[0]), torch.zeros(self.max_length-ur[1])], dim=0))
        data_info['utterance_mask'] = torch.stack(utterance_mask, dim=0)
        data_info['label'] = torch.tensor(self.features[item].label, dtype=torch.float)

        return data_info

    def __len__(self):
        return len(self.features)


def collate_fn(data):
    qid = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    p_mask = []
    context = []
    key_utterance = []
    question_mask = []
    utterance_mask = []
    label = []
    label_mask = []
    for d in data:
        qid.append(d['qid'])
        input_ids.append(d['input_ids'])
        attention_mask.append(d['attention_mask'])
        token_type_ids.append(d['token_type_ids'])
        p_mask.append(d['p_mask'])
        context.append(d['context'])
        key_utterance.append(d['key_utterance'])
        question_mask.append(d['question_mask'])
        utterance_mask.append(d['utterance_mask'])
        label.append(d['label'])
        label_mask.append(torch.ones_like(d['label']))
    data_info = dict()
    data_info['qid'] = qid
    data_info['input_ids'] = torch.stack(input_ids, dim=0).cuda()
    data_info['attention_mask'] = torch.stack(attention_mask, dim=0).cuda()
    data_info['token_type_ids'] = torch.stack(token_type_ids, dim=0).cuda()
    data_info['p_mask'] = torch.stack(p_mask, dim=0).cuda()
    data_info['context'] = context
    data_info['key_utterance'] = key_utterance
    data_info['question_mask'] = pad_sequence(question_mask, batch_first=True, padding_value=0).cuda()
    data_info['utterance_mask'] = pad_sequence(utterance_mask, batch_first=True, padding_value=0).cuda()
    data_info['label'] = pad_sequence(label, batch_first=True, padding_value=0).cuda()
    data_info['label_mask'] = pad_sequence(label_mask, batch_first=True, padding_value=0).cuda()

    return data_info


def get_examples(input_file, tokenizer):
    gather_token = tokenizer.sep_token

    def _get_pruned_context(start_uid: int, utterances: list):
        max_context_length = args.max_length - args.question_max_length
        utterance_start_poses, speaker_start_poses = [], []
        context = ''
        for idx in range(start_uid, len(utterances)):
            utter_dict = utterances[idx]
            speaker = ' & '.join(utter_dict['speakers'])
            text = utter_dict['utterance']
            tmp_context = context + gather_token + ' ' + speaker + ' : ' + text + ' '
            cur_len = len(tokenizer.encode(tmp_context.strip()[len(gather_token + ' '):])) - 1
            # plus 1 for later adding pad token in the front, additional 1 for the final [CLS] token if we use cls as gather token
            if cur_len > max_context_length:
                return context, idx, utterance_start_poses, speaker_start_poses
            context += gather_token + ' '
            speaker_start_poses.append(len(context))
            context += speaker + ' : '
            utterance_start_poses.append(len(context))
            context += text + ' '
        return context, len(utterances), utterance_start_poses, speaker_start_poses

    examples = []
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']

    max_utter_num, number_before_pruning = 0, 0
    for scene in tqdm(input_data):
        paragraphs = scene['paragraphs']
        for dialogue in paragraphs:
            relations = None
            utterances = dialogue['utterances:']
            utterances_cased = dialogue['utterances:']
            max_utter_num = max(max_utter_num, len(utterances))
            for idx in range(len(utterances)):
                utterances[idx]['utterance'] = utterances[idx]['utterance'].lower()
                for sidx in range(len(utterances[idx]['speakers'])):
                    utterances[idx]['speakers'][sidx] = utterances[idx]['speakers'][sidx].lower()
            for idx in range(len(dialogue['qas'])):
                dialogue['qas'][idx]['question'] = dialogue['qas'][idx]['question'].lower()
                for aidx in range(len(dialogue['qas'][idx]['answers'])):
                    dialogue['qas'][idx]['answers'][aidx]['answer_text'] = dialogue['qas'][idx]['answers'][aidx]['answer_text'].lower()

            idx = 0
            context_dict_list = []
            while idx < len(utterances):
                context, next_idx, utterance_start_poses, speaker_start_poses = _get_pruned_context(idx, utterances)
                context = context.strip()[len(gather_token + ' '):]  # remove the first gather token and ' '
                # context = tokenizer.pad_token + ' ' + context

                # modified_offset = len(tokenizer.pad_token + ' ') - len(gather_token + ' ')
                modified_offset = - len(gather_token + ' ')
                utterance_start_poses = [pos + modified_offset for pos in utterance_start_poses]
                speaker_start_poses = [pos + modified_offset for pos in speaker_start_poses]
                context_dict_list.append(
                    {'start_uid': idx,
                     'end_uid': next_idx - 1,
                     'context': context,
                     'utterance_start_poses': utterance_start_poses,
                     'speaker_start_poses': speaker_start_poses,
                     'utterances': utterances[idx: next_idx],
                     'utterances_cased': utterances_cased[idx: next_idx]
                     }
                )
                for ii in range(next_idx - idx):
                    a = context[utterance_start_poses[ii]: utterance_start_poses[ii] + len(utterances[ii + idx]['utterance'])]
                    b = utterances[ii + idx]['utterance']
                    assert a == b, "{} vs. {}".format(a, b)
                idx = next_idx
            assert idx == len(utterances)

            for qa in dialogue['qas']:
                question = qa['question']
                qid = qa['id']

                answer_utter_id_list = set()

                for answer in qa['answers']:
                    answer_utter_id = answer['utterance_id']
                    answer_utter_id_list.add(answer_utter_id)

                answer_utter_id_list = list(answer_utter_id_list)

                for idx, context_dict in enumerate(context_dict_list):
                    context_dict_key_utter_list = []
                    for answer_utter in answer_utter_id_list:
                        if answer_utter in range(context_dict['start_uid'], context_dict['end_uid'] + 1):
                            context_dict_key_utter_list.append(answer_utter - context_dict['start_uid'])

                    exp = ConvExample(context=context_dict['context'], utterances=context_dict['utterances'], question=question, qid=qid+'-'+str(idx),
                                      key_utterance=context_dict_key_utter_list, utterances_cased=context_dict['utterances_cased'])
                    examples.append(exp)
    return examples


class EntFeature:
    def __init__(self, sqc_input_ids, sqc_attention_mask, sqc_token_type_ids,
                 sq_input_ids, sq_attention_mask, sq_token_type_ids,
                 speaker, mapped_speaker, qid, label, label1, label2=None):
        self.sqc_input_ids = sqc_input_ids
        self.sqc_attention_mask = sqc_attention_mask
        self.sqc_token_type_ids = sqc_token_type_ids
        self.sq_input_ids = sq_input_ids
        self.sq_attention_mask = sq_attention_mask
        self.sq_token_type_ids = sq_token_type_ids
        self.speaker = speaker
        self.mapped_speaker = mapped_speaker
        self.qid = qid
        self.label = label
        self.label1 = label1
        self.label2 = label2


class EntDataset(Dataset):
    def __init__(self, features):
        super(EntDataset, self).__init__()
        self.features = features

    def __getitem__(self, item):
        data_info = dict()
        data_info['sqc_input_ids'] = torch.tensor(self.features[item].sqc_input_ids, dtype=torch.long)
        data_info['sqc_attention_mask'] = torch.tensor(self.features[item].sqc_attention_mask, dtype=torch.float)
        data_info['sqc_token_type_ids'] = torch.tensor(self.features[item].sqc_token_type_ids, dtype=torch.long)
        data_info['sq_input_ids'] = torch.tensor(self.features[item].sq_input_ids, dtype=torch.long)
        data_info['sq_attention_mask'] = torch.tensor(self.features[item].sq_attention_mask, dtype=torch.float)
        data_info['sq_token_type_ids'] = torch.tensor(self.features[item].sq_token_type_ids, dtype=torch.long)
        data_info['speaker'] = self.features[item].speaker
        data_info['mapped_speaker'] = self.features[item].mapped_speaker
        data_info['qid'] = self.features[item].qid
        data_info['label'] = torch.tensor(self.features[item].label, dtype=torch.float)
        data_info['label1'] = torch.tensor(self.features[item].label1, dtype=torch.long)

        return data_info

    def __len__(self):
        return len(self.features)


def ent_collate_fn(data):
    sqc_input_ids = []
    sqc_attention_mask = []
    sqc_token_type_ids = []
    sq_input_ids = []
    sq_attention_mask = []
    sq_token_type_ids = []
    speaker = []
    mapped_speaker = []
    qid = []
    label = []
    label1 = []
    for d in data:
        sqc_input_ids.append(d['sqc_input_ids'])
        sqc_attention_mask.append(d['sqc_attention_mask'])
        sqc_token_type_ids.append(d['sqc_token_type_ids'])
        sq_input_ids.append(d['sq_input_ids'])
        sq_attention_mask.append(d['sq_attention_mask'])
        sq_token_type_ids.append(d['sq_token_type_ids'])
        speaker.append(d['speaker'])
        mapped_speaker.append(d['mapped_speaker'])
        qid.append(d['qid'])
        label.append(d['label'])
        label1.append(d['label1'])
    # sqc_input_ids = torch.stack(sqc_input_ids, dim=0).cuda()
    # sqc_attention_mask = torch.stack(sqc_attention_mask, dim=0).cuda()
    # sqc_token_type_ids = torch.stack(sqc_token_type_ids, dim=0).cuda()
    #
    sqc_input_ids = pad_sequence(sqc_input_ids, batch_first=True, padding_value=0).cuda()
    sqc_attention_mask = pad_sequence(sqc_attention_mask, batch_first=True, padding_value=0).cuda()
    sqc_token_type_ids = pad_sequence(sqc_token_type_ids, batch_first=True, padding_value=0).cuda()

    sq_input_ids = pad_sequence(sq_input_ids, batch_first=True, padding_value=0).cuda()
    sq_attention_mask = pad_sequence(sq_attention_mask, batch_first=True, padding_value=0).cuda()
    sq_token_type_ids = pad_sequence(sq_token_type_ids, batch_first=True, padding_value=0).cuda()
    label = torch.stack(label, dim=0).cuda()
    data_info = dict()
    data_info['sqc_input_ids'] = sqc_input_ids
    data_info['sqc_attention_mask'] = sqc_attention_mask
    data_info['sqc_token_type_ids'] = sqc_token_type_ids
    data_info['sq_input_ids'] = sq_input_ids
    data_info['sq_attention_mask'] = sq_attention_mask
    data_info['sq_token_type_ids'] = sq_token_type_ids
    data_info['speaker'] = speaker
    data_info['mapped_speaker'] = mapped_speaker
    data_info['qid'] = qid
    data_info['label'] = label
    data_info['label1'] = label1

    return data_info


def get_speaker_key_features(examples, tokenizer, max_length, entities, speaker_mapping):
    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id,
                  tokenizer.bos_token_id, tokenizer.cls_token_id,
                  tokenizer.pad_token_id]
    gather_token = tokenizer.sep_token
    # nlp_pipeline = stanza.Pipeline('en')
    features = []

    question_dict = {}
    num_count = 0
    dist_count = {}

    for fidx, exp in enumerate(tqdm(examples)):
        speaker_counts = set()
        for utter_dict in exp.utterances:
            for speaker in utter_dict['speakers']:
                speaker_counts.add(speaker)
        speaker_counts = list(speaker_counts)
        qid = exp.qid
        qid_ori = qid.split('-')[0]
        entity = entities[qid_ori]
        speaker_list = []
        if entity is not None:
            for e in entity:
                if e.type == 'PERSON':
                    speaker_list.append(e.text)
        context = ''
        context_dict = dict()
        speaker_corr_utter = dict()
        for spk in speaker_counts:
            speaker_corr_utter[spk] = []
            if exp.utterances[0]['speakers'][0] == '#note#':
                context_dict[spk] = gather_token + ' #note# : ' + exp.utterances[0]['utterance'] + ' '
            else:
                context_dict[spk] = ''
        question = exp.question
        for uidx, utter_dict in enumerate(exp.utterances):
            text = utter_dict['utterance']
            speaker = ' & '.join(utter_dict['speakers'])
            context += gather_token + ' ' + speaker + ' : ' + text + ' '
            for s in utter_dict['speakers']:
                if s in context_dict:
                    context_dict[s] += gather_token + ' ' + s + ' : ' + text + ' '
                if s in speaker_corr_utter:
                    speaker_corr_utter[s].append(uidx)
        for k, v in context_dict.items():
            context_dict[k] = v.strip()[len(gather_token) + 1:]

        context = context.strip()[len(gather_token) + 1:]  # remove the first sep token and ' '

        assert context == exp.context

        key_utterance = exp.key_utterance

        if len(key_utterance) > 0:
            for s in speaker_list:
                if s in speaker_mapping:
                    mapped_speakers = speaker_mapping[s]
                    s = s.lower()
                    related_speaker = set()
                    for k in key_utterance:
                        for spk in exp.utterances[k]['speakers']:
                            related_speaker.add(spk)
                    if len(mapped_speakers) > 0:
                        if type(mapped_speakers[0]) == list:
                            for m_s in mapped_speakers[0]:
                                m_s = m_s.lower()
                                if m_s in speaker_counts:
                                    if m_s in list(related_speaker):
                                        speaker_label = 1
                                        speaker_label1 = 1
                                        speaker_label2 = 1
                                    else:
                                        # if exp.utterances[0]['speakers'] == "#note#" and "#note#" in related_speaker:
                                        #     if s in exp.utterances[0]['utterance']:
                                        #         speaker_label = 1
                                        #     else:
                                        #         speaker_label = 0
                                        # else:
                                        # speaker_label = 0
                                        speaker_label = 0
                                        key_utter = key_utterance[0]
                                        m_s_utter = speaker_corr_utter[m_s]
                                        ii = 0
                                        for s_u in m_s_utter:
                                            if key_utter < s_u:
                                                break
                                            ii += 1
                                        if ii == 0:
                                            # print('appear before the speaker')
                                            speaker_label1 = 2
                                            s_u_i = m_s_utter[ii]
                                            dist = s_u_i - key_utter
                                            if dist in dist_count:
                                                dist_count[dist] += 1
                                            else:
                                                dist_count[dist] = 1
                                            # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                            # doc = nlp_pipeline(utterance_i)
                                            # speaker_shift = False
                                            # if len(doc.sentences[0].ents) > 0:
                                            #     for ent in doc.sentences[0].ents:
                                            #         if ent.type == 'PERSON':
                                            #             sss = ent.text
                                            #             if sss in speaker_mapping:
                                            #                 sss_mapped = speaker_mapping[sss]
                                            #                 for s_mp in sss_mapped:
                                            #                     s_mp = s_mp.lower()
                                            #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                            #                         # print('before: speaker shift')
                                            #                         speaker_label2 = 2
                                            #                         speaker_shift = True
                                            #                         break
                                            #             if speaker_shift:
                                            #                 break
                                            # if not speaker_shift:
                                            #     # print('before: no speaker shift')
                                            #     speaker_label2 = 3
                                        elif ii == len(m_s_utter):
                                            # print('appear after the speaker')
                                            speaker_label1 = 3
                                            s_u_i = m_s_utter[ii-1]
                                            dist = s_u_i - key_utter
                                            if dist in dist_count:
                                                dist_count[dist] += 1
                                            else:
                                                dist_count[dist] = 1
                                            # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                            # doc = nlp_pipeline(utterance_i)
                                            # speaker_shift = False
                                            # if len(doc.sentences[0].ents) > 0:
                                            #     for ent in doc.sentences[0].ents:
                                            #         if ent.type == 'PERSON':
                                            #             sss = ent.text
                                            #             if sss in speaker_mapping:
                                            #                 sss_mapped = speaker_mapping[sss]
                                            #                 for s_mp in sss_mapped:
                                            #                     s_mp = s_mp.lower()
                                            #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                            #                         # print('after: speaker shift')
                                            #                         speaker_label2 = 4
                                            #                         speaker_shift = True
                                            #                         break
                                            #             if speaker_shift:
                                            #                 break
                                            # if not speaker_shift:
                                            #     speaker_label2 = 5
                                            #     # print('after: no speaker shift')
                                        else:
                                            s_u_i = m_s_utter[ii]
                                            s_u_i_1 = m_s_utter[ii-1]
                                            d_i = s_u_i - key_utter
                                            d_i_1 = key_utter - s_u_i_1
                                            if d_i_1 <= d_i:
                                                # print('appear after the speaker')
                                                speaker_label1 = 3
                                                dist = s_u_i_1 - key_utter
                                                if dist in dist_count:
                                                    dist_count[dist] += 1
                                                else:
                                                    dist_count[dist] = 1
                                                # utterance_i = exp.utterances_cased[s_u_i_1]['utterance']
                                                # doc = nlp_pipeline(utterance_i)
                                                # speaker_shift = False
                                                # if len(doc.sentences[0].ents) > 0:
                                                #     for ent in doc.sentences[0].ents:
                                                #         if ent.type == 'PERSON':
                                                #             sss = ent.text
                                                #             if sss in speaker_mapping:
                                                #                 sss_mapped = speaker_mapping[sss]
                                                #                 for s_mp in sss_mapped:
                                                #                     s_mp = s_mp.lower()
                                                #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                                #                         # print('after: speaker shift')
                                                #                         speaker_shift = True
                                                #                         speaker_label2 = 4
                                                #                         break
                                                #             if speaker_shift:
                                                #                 break
                                                # if not speaker_shift:
                                                #     speaker_label2 = 5
                                                #     # print('after: no speaker shift')
                                            else:
                                                # print('appear before the speaker')
                                                speaker_label1 = 2
                                                dist = s_u_i - key_utter
                                                if dist in dist_count:
                                                    dist_count[dist] += 1
                                                else:
                                                    dist_count[dist] = 1
                                                # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                                # doc = nlp_pipeline(utterance_i)
                                                # speaker_shift = False
                                                # if len(doc.sentences[0].ents) > 0:
                                                #     for ent in doc.sentences[0].ents:
                                                #         if ent.type == 'PERSON':
                                                #             sss = ent.text
                                                #             if sss in speaker_mapping:
                                                #                 sss_mapped = speaker_mapping[sss]
                                                #                 for s_mp in sss_mapped:
                                                #                     s_mp = s_mp.lower()
                                                #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                                #                         # print('before: speaker shift')
                                                #                         speaker_shift = True
                                                #                         speaker_label2 = 2
                                                #                         break
                                                #             if speaker_shift:
                                                #                 break
                                                # if not speaker_shift:
                                                #     speaker_label2 = 3
                                                #     # print('before: no speaker shift')
                                    spk_qid = qid + '-' + s + '-' + m_s

                                    speaker_token = s + ' ' + tokenizer.sep_token + ' ' + m_s

                                    sq_ids_dict = tokenizer.encode_plus(speaker_token, question,
                                                                        truncation=True, max_length=max_length,
                                                                        return_offsets_mapping=True)

                                    sq_input_ids = sq_ids_dict['input_ids']
                                    sq_attention_mask = sq_ids_dict['attention_mask']
                                    sq_token_type_ids = sq_ids_dict['token_type_ids']

                                    # padding question
                                    # question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                                    # if question_length > args.question_max_length:
                                    #     while len(tokenizer.encode(question)) > args.question_max_length:
                                    #         question = question[:-1]
                                    # remain_length_q = args.question_max_length - question_length
                                    # question += ' '.join([tokenizer.pad_token] * remain_length_q)

                                    speaker_question = speaker_token + ' ' + tokenizer.sep_token + ' ' + question
                                    spk_q_len = len(tokenizer.encode(speaker_question))  # including [CLS] and [SEP]

                                    # padding context
                                    # context_max_length = args.max_length - spk_q_len
                                    # context_length = len(tokenizer.encode(context)) - 1  # except the [CLS] and including the [SEP]
                                    # if context_length > context_max_length:
                                    #     while (len(tokenizer.encode(context)) - 1) > context_max_length:
                                    #         context = context[:-1]
                                    # remain_length = context_max_length - context_length
                                    # context += ' '.join([tokenizer.pad_token] * remain_length)
                                    # assert len(tokenizer.encode(context)) >= context_max_length

                                    if m_s not in context_dict:
                                        print(context_dict.keys())
                                        print(m_s)
                                        print(question)
                                    context_spk = context_dict[m_s]

                                    question_context = question + ' ' + tokenizer.sep_token + ' ' + context_spk

                                    ids_dict = tokenizer.encode_plus(speaker_token, question_context,
                                                                     truncation=True, max_length=max_length,
                                                                     return_offsets_mapping=True)
                                    sqc_input_ids = ids_dict['input_ids']
                                    sqc_attention_mask = ids_dict['attention_mask']
                                    sqc_token_type_ids = ids_dict['token_type_ids']

                                    feature = EntFeature(sqc_input_ids=sqc_input_ids, sqc_attention_mask=sqc_attention_mask,
                                                         sqc_token_type_ids=sqc_token_type_ids, sq_input_ids=sq_input_ids,
                                                         sq_attention_mask=sq_attention_mask, sq_token_type_ids=sq_token_type_ids,
                                                         speaker=s, mapped_speaker=m_s, qid=spk_qid, label=speaker_label,
                                                         label1=speaker_label1)
                                    features.append(feature)
                                    num_count += 1
                                    if qid_ori in question_dict:
                                        question_dict[qid_ori] += 1
                                    else:
                                        question_dict[qid_ori] = 1

                        else:
                            for m_s in mapped_speakers:
                                m_s = m_s.lower()
                                if m_s in speaker_counts:
                                    if m_s in list(related_speaker):
                                        speaker_label = 1
                                        speaker_label1 = 1
                                        speaker_label2 = 1
                                    else:
                                        # if exp.utterances[0]['speakers'] == "#note#" and "#note#" in related_speaker:
                                        #     if s in exp.utterances[0]['utterance']:
                                        #         speaker_label = 1
                                        #     else:
                                        #         speaker_label = 0
                                        # else:
                                        speaker_label = 0
                                        key_utter = key_utterance[0]
                                        m_s_utter = speaker_corr_utter[m_s]
                                        ii = 0
                                        for s_u in m_s_utter:
                                            if key_utter < s_u:
                                                break
                                            ii += 1
                                        if ii == 0:
                                            # print('appear before the speaker')
                                            speaker_label1 = 2
                                            s_u_i = m_s_utter[ii]
                                            dist = s_u_i - key_utter
                                            if dist in dist_count:
                                                dist_count[dist] += 1
                                            else:
                                                dist_count[dist] = 1
                                            # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                            # doc = nlp_pipeline(utterance_i)
                                            # speaker_shift = False
                                            # if len(doc.sentences[0].ents) > 0:
                                            #     for ent in doc.sentences[0].ents:
                                            #         if ent.type == 'PERSON':
                                            #             sss = ent.text
                                            #             if sss in speaker_mapping:
                                            #                 sss_mapped = speaker_mapping[sss]
                                            #                 for s_mp in sss_mapped:
                                            #                     s_mp = s_mp.lower()
                                            #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                            #                         # print('before: speaker shift')
                                            #                         speaker_label2 = 2
                                            #                         speaker_shift = True
                                            #                         break
                                            #             if speaker_shift:
                                            #                 break
                                            # if not speaker_shift:
                                            #     speaker_label2 = 3
                                            #     # print('before: no speaker shift')
                                        elif ii == len(m_s_utter):
                                            # print('appear after the speaker')
                                            speaker_label1 = 3
                                            s_u_i = m_s_utter[ii - 1]
                                            dist = s_u_i - key_utter
                                            if dist in dist_count:
                                                dist_count[dist] += 1
                                            else:
                                                dist_count[dist] = 1
                                            # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                            # doc = nlp_pipeline(utterance_i)
                                            # speaker_shift = False
                                            # if len(doc.sentences[0].ents) > 0:
                                            #     for ent in doc.sentences[0].ents:
                                            #         if ent.type == 'PERSON':
                                            #             sss = ent.text
                                            #             if sss in speaker_mapping:
                                            #                 sss_mapped = speaker_mapping[sss]
                                            #                 for s_mp in sss_mapped:
                                            #                     s_mp = s_mp.lower()
                                            #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                            #                         # print('after: speaker shift')
                                            #                         speaker_shift = True
                                            #                         speaker_label2 = 4
                                            #                         break
                                            #             if speaker_shift:
                                            #                 break
                                            # if not speaker_shift:
                                            #     speaker_label2 = 5
                                            #     # print('after: no speaker shift')
                                        else:
                                            s_u_i = m_s_utter[ii]
                                            s_u_i_1 = m_s_utter[ii - 1]
                                            d_i = s_u_i - key_utter
                                            d_i_1 = key_utter - s_u_i_1
                                            if d_i_1 <= d_i:
                                                # print('appear after the speaker')
                                                speaker_label1 = 3
                                                dist = s_u_i_1 - key_utter
                                                if dist in dist_count:
                                                    dist_count[dist] += 1
                                                else:
                                                    dist_count[dist] = 1
                                                # utterance_i = exp.utterances_cased[s_u_i_1]['utterance']
                                                # doc = nlp_pipeline(utterance_i)
                                                # speaker_shift = False
                                                # if len(doc.sentences[0].ents) > 0:
                                                #     for ent in doc.sentences[0].ents:
                                                #         if ent.type == 'PERSON':
                                                #             sss = ent.text
                                                #             if sss in speaker_mapping:
                                                #                 sss_mapped = speaker_mapping[sss]
                                                #                 for s_mp in sss_mapped:
                                                #                     s_mp = s_mp.lower()
                                                #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                                #                         # print('after: speaker shift')
                                                #                         speaker_shift = True
                                                #                         speaker_label2 = 4
                                                #                         break
                                                #             if speaker_shift:
                                                #                 break
                                                # if not speaker_shift:
                                                #     speaker_label2 = 5
                                                #     # print('after: no speaker shift')
                                            else:
                                                # print('appear before the speaker')
                                                speaker_label1 = 2
                                                dist = s_u_i - key_utter
                                                if dist in dist_count:
                                                    dist_count[dist] += 1
                                                else:
                                                    dist_count[dist] = 1
                                                # utterance_i = exp.utterances_cased[s_u_i]['utterance']
                                                # doc = nlp_pipeline(utterance_i)
                                                # speaker_shift = False
                                                # if len(doc.sentences[0].ents) > 0:
                                                #     for ent in doc.sentences[0].ents:
                                                #         if ent.type == 'PERSON':
                                                #             sss = ent.text
                                                #             if sss in speaker_mapping:
                                                #                 sss_mapped = speaker_mapping[sss]
                                                #                 for s_mp in sss_mapped:
                                                #                     s_mp = s_mp.lower()
                                                #                     if s_mp in exp.utterances[key_utter]['speakers']:
                                                #                         # print('before: speaker shift')
                                                #                         speaker_shift = True
                                                #                         speaker_label2 = 2
                                                #                         break
                                                #             if speaker_shift:
                                                #                 break
                                                # if not speaker_shift:
                                                #     speaker_label2 = 3
                                                #     # print('before: no speaker shift')
                                    spk_qid = qid + '-' + s + '-' + m_s

                                    speaker_token = s + ' ' + tokenizer.sep_token + ' ' + m_s

                                    sq_ids_dict = tokenizer.encode_plus(speaker_token, question,
                                                                        truncation=True, max_length=max_length,
                                                                        return_offsets_mapping=True)

                                    sq_input_ids = sq_ids_dict['input_ids']
                                    sq_attention_mask = sq_ids_dict['attention_mask']
                                    sq_token_type_ids = sq_ids_dict['token_type_ids']

                                    # padding question
                                    # question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                                    # if question_length > args.question_max_length:
                                    #     while len(tokenizer.encode(question)) > args.question_max_length:
                                    #         question = question[:-1]
                                    # remain_length_q = args.question_max_length - question_length
                                    # question += ' '.join([tokenizer.pad_token] * remain_length_q)

                                    speaker_question = speaker_token + ' ' + tokenizer.sep_token + ' ' + question
                                    spk_q_len = len(tokenizer.encode(speaker_question))  # including [CLS] and [SEP]

                                    # padding context
                                    # context_max_length = args.max_length - spk_q_len
                                    # context_length = len(tokenizer.encode(context)) - 1  # except the [CLS] and including the [SEP]
                                    # if context_length > context_max_length:
                                    #     while (len(tokenizer.encode(context)) - 1) > context_max_length:
                                    #         context = context[:-1]
                                    # remain_length = context_max_length - context_length
                                    # context += ' '.join([tokenizer.pad_token] * remain_length)
                                    # assert len(tokenizer.encode(context)) >= context_max_length

                                    context_spk = context_dict[m_s]

                                    question_context = question + ' ' + tokenizer.sep_token + ' ' + context_spk

                                    ids_dict = tokenizer.encode_plus(speaker_token, question_context,
                                                                     truncation=True, max_length=max_length,
                                                                     return_offsets_mapping=True)
                                    sqc_input_ids = ids_dict['input_ids']
                                    sqc_attention_mask = ids_dict['attention_mask']
                                    sqc_token_type_ids = ids_dict['token_type_ids']

                                    feature = EntFeature(sqc_input_ids=sqc_input_ids, sqc_attention_mask=sqc_attention_mask,
                                                         sqc_token_type_ids=sqc_token_type_ids, sq_input_ids=sq_input_ids,
                                                         sq_attention_mask=sq_attention_mask, sq_token_type_ids=sq_token_type_ids,
                                                         speaker=s, mapped_speaker=m_s, qid=spk_qid, label=speaker_label,
                                                         label1=speaker_label1)
                                    features.append(feature)
                                    num_count += 1
                                    if qid_ori in question_dict:
                                        question_dict[qid_ori] += 1
                                    else:
                                        question_dict[qid_ori] = 1
                    # for m_s in mapped_speakers:
                    #     if m_s in list(related_speaker):
                    #         speaker_label = 1
                    #     else:
                    #         if exp.utterances[0]['speakers'] == "#note#" and "#note#" in related_speaker:
                    #             if s in exp.utterances[0]['utterance']:
                    #                 speaker_label = 1
                    #             else:
                    #                 speaker_label = 0
                    #         else:
                    #             speaker_label = 0
                    #     spk_qid = qid + '-' + s + '-' + m_s
                    #
                    #     speaker_token = s + ' ' + tokenizer.sep_token + ' ' + m_s
                    #
                    #     sq_ids_dict = tokenizer.encode_plus(speaker_token, question, padding='max_length',
                    #                                         truncation=True, max_length=max_length,
                    #                                         return_offsets_mapping=True)
                    #
                    #     sq_input_ids = sq_ids_dict['input_ids']
                    #     sq_attention_mask = sq_ids_dict['attention_mask']
                    #     sq_token_type_ids = sq_ids_dict['token_type_ids']
                    #
                    #     # padding question
                    #     question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                    #     if question_length > args.question_max_length:
                    #         while len(tokenizer.encode(question)) > args.question_max_length:
                    #             question = question[:-1]
                    #     remain_length_q = args.question_max_length - question_length
                    #     question += ' '.join([tokenizer.pad_token] * remain_length_q)
                    #
                    #     speaker_question = speaker_token + ' ' + tokenizer.sep_token + ' ' + question
                    #     spk_q_len = len(tokenizer.encode(speaker_question))    # including [CLS] and [SEP]
                    #
                    #     # padding context
                    #     context_max_length = args.max_length - spk_q_len
                    #     context_length = len(tokenizer.encode(context)) - 1  # except the [CLS] and including the [SEP]
                    #     if context_length > context_max_length:
                    #         while (len(tokenizer.encode(context)) - 1) > context_max_length:
                    #             context = context[:-1]
                    #     remain_length = context_max_length - context_length
                    #     context += ' '.join([tokenizer.pad_token] * remain_length)
                    #     assert len(tokenizer.encode(context)) >= context_max_length
                    #
                    #     question_context = question + ' ' + tokenizer.sep_token + ' ' + context
                    #
                    #     ids_dict = tokenizer.encode_plus(speaker_token, question_context, padding='max_length',
                    #                                      truncation=True, max_length=max_length,
                    #                                      return_offsets_mapping=True)
                    #     sqc_input_ids = ids_dict['input_ids']
                    #     sqc_attention_mask = ids_dict['attention_mask']
                    #     sqc_token_type_ids = ids_dict['token_type_ids']
                    #
                    #     feature = EntFeature(sqc_input_ids=sqc_input_ids, sqc_attention_mask=sqc_attention_mask,
                    #                          sqc_token_type_ids=sqc_token_type_ids, sq_input_ids=sq_input_ids,
                    #                          sq_attention_mask=sq_attention_mask, sq_token_type_ids=sq_token_type_ids,
                    #                          speaker=s, mapped_speaker=m_s, qid=spk_qid, label=speaker_label)
                    #     features.append(feature)
    print(len(question_dict))
    print(num_count)
    print(dist_count)
    return features, question_dict


def get_ent_dataset(input_file, cache_path, tokenizer, max_length, entity_file, speaker_mapping_file):
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    postfix = ""
    for type_ in ["trn", "dev", "tst"]:
        if type_ in input_file:
            postfix = type_
            break
    example_path = os.path.join(cache_path, "example_{}_ent.pkl".format(postfix))
    print("Read {}_examples from cache...".format(postfix))
    if not os.path.exists(example_path):
        examples = get_examples(input_file, tokenizer)
        pickle.dump(examples, open(example_path, 'wb'))
    else:
        examples = pickle.load(open(example_path, 'rb'), encoding='utf-8')

    feature_path = os.path.join(cache_path, "ent_features_{}_2.pkl".format(postfix))
    entities = pickle.load(open(entity_file, 'rb'), encoding='utf-8')
    speaker_mapping = json.load(open(speaker_mapping_file, 'r', encoding='utf-8'))
    if not os.path.exists(feature_path):
        features, _ = get_speaker_key_features(examples, tokenizer, max_length, entities, speaker_mapping)
        pickle.dump(features, open(feature_path, 'wb'))
    else:
        print("Read features from cache...".format(postfix))
        features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')
    dataset = EntDataset(features)
    return dataset


def get_ent_data_loader(train_file, dev_file, test_file, cache_path, tokenizer, max_length, shuffle=True):
    train_set = get_ent_dataset(train_file, cache_path, tokenizer, max_length, 'data/friendsqa_trn_ent.pkl', 'speaker_map/trn_mapping.json')
    dev_set = get_ent_dataset(dev_file, cache_path, tokenizer, max_length, 'data/friendsqa_dev_ent.pkl', 'speaker_map/dev_mapping.json')
    test_set = get_ent_dataset(test_file, cache_path, tokenizer, max_length, 'data/friendsqa_tst_ent.pkl', 'speaker_map/tst_mapping.json')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=ent_collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=ent_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=ent_collate_fn)

    return train_loader, dev_loader, test_loader


def get_key_features(examples, tokenizer, max_length):
    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id,
                  tokenizer.bos_token_id, tokenizer.cls_token_id,
                  tokenizer.pad_token_id]
    gather_token = tokenizer.sep_token
    features = []

    for fidx, exp in enumerate(tqdm(examples)):
        question = exp.question
        context = ''
        for uidx, utter_dict in enumerate(exp.utterances):
            text = utter_dict['utterance']
            speaker = ' & '.join(utter_dict['speakers'])
            context += gather_token + ' ' + speaker + ' : ' + text + ' '

        context = context.strip()[len(gather_token) + 1:]  # remove the first sep token and ' '

        assert context == exp.context

        # padding context
        context_max_length = args.max_length - args.question_max_length
        context_length = len(tokenizer.encode(context)) - 1    # except the [CLS] and including the [SEP]
        remain_length = context_max_length - context_length
        context += ' '.join([tokenizer.pad_token] * remain_length)
        assert len(tokenizer.encode(context)) >= context_max_length

        # padding question
        question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
        if question_length > args.question_max_length:
            while len(tokenizer.encode(question)) > args.question_max_length:
                question = question[:-1]
        remain_length_q = args.question_max_length - question_length
        question += ' '.join([tokenizer.pad_token] * remain_length_q)

        ids_dict = tokenizer.encode_plus(question, context, padding='max_length',
                                         truncation=True, max_length=max_length,
                                         return_offsets_mapping=True)
        input_ids = ids_dict['input_ids']
        attention_mask = ids_dict['attention_mask']
        token_type_ids = ids_dict['token_type_ids']
        for i in range(len(attention_mask)):
            if input_ids[i] == tokenizer.pad_token_id:
                attention_mask[i] = 0
        p_mask = [1] * len(input_ids)
        all_utterance_start_pos = []
        for i in range(len(input_ids)):
            if input_ids[i] in p_mask_ids or token_type_ids[i] == 0:
                p_mask[i] = 0
            if input_ids[i] == tokenizer.sep_token_id:
                if i == len(input_ids) - 1:
                    continue
                else:
                    all_utterance_start_pos.append(i)

        label = []
        tri_range = []
        key_utterance = exp.key_utterance
        num_utter = len(exp.utterances)
        if num_utter <= 3:
            for i in range(num_utter):
                if i in key_utterance:
                    label.append(1)
                    break
                else:
                    if i == num_utter - 1:
                        label.append(0)
            tri_range.append([all_utterance_start_pos[0], args.max_length - remain_length])
        else:
            for i in range(num_utter-2):
                max_i = i + 3
                if max_i == num_utter:
                    tri_range.append([all_utterance_start_pos[i], args.max_length - remain_length])
                else:
                    tri_range.append([all_utterance_start_pos[i], all_utterance_start_pos[max_i]])
                contain_key = False
                for j in range(i, max_i):
                    if j in key_utterance:
                        contain_key = True
                        break
                if contain_key:
                    label.append(1)
                else:
                    label.append(0)
        f_tmp = ConvFeature(qid=exp.qid, input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, p_mask=p_mask,
                            context=context, key_utterance=key_utterance, question_len=args.question_max_length-remain_length_q-1,
                            utterance_range=tri_range, label=label)
        features.append(f_tmp)

    return features


def get_dataset(input_file, cache_path, tokenizer, max_length):
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    postfix = ""
    for type_ in ["trn", "dev", "tst"]:
        if type_ in input_file:
            postfix = type_
            break
    example_path = os.path.join(cache_path, "example_{}_key.pkl".format(postfix))
    if not os.path.exists(example_path):
        examples = get_examples(input_file, tokenizer)
        pickle.dump(examples, open(example_path, 'wb'))
        print('Examples saved to' + example_path)
    else:
        print("Read {}_examples from cache...".format(postfix))
        examples = pickle.load(open(example_path, 'rb'), encoding='utf-8')

    feature_path = os.path.join(cache_path, "features_{}_key.pkl".format(postfix))
    if not os.path.exists(feature_path):
        features = get_key_features(examples, tokenizer, max_length)
        pickle.dump(features, open(feature_path, 'wb'))
    else:
        print("Read features from cache...".format(postfix))
        features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')

    dataset = ConvDataset(features)
    return dataset


def get_dataloader(train_file, dev_file, test_file, cache_path, tokenizer, max_length, shuffle=True):
    train_set = get_dataset(train_file, cache_path, tokenizer, max_length)
    dev_set = get_dataset(dev_file, cache_path, tokenizer, max_length)
    test_set = get_dataset(test_file, cache_path, tokenizer, max_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader
