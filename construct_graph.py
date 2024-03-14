import torch
from new_utils_2 import Example, InputFeature, Dataset
from transformers import ElectraTokenizerFast
import pickle
import tqdm
import torch
import json
import stanza
# import spacy
# import neurealcoref


def match(seq1, seq2, begin_idx=0):
    gets = False
    q_start_pos = None
    for i in range(len(seq1[begin_idx:]) - len(seq2)):
        for j in range(len(seq2)):
            if seq2[j] == seq1[begin_idx + i + j]:
                if j == len(seq2) - 1:
                    gets = True
            else:
                break
        if gets:
            q_start_pos = i + begin_idx
            break
    return q_start_pos


def get_data(data_path, feature_path, save_path, context_range=1, n_gram=1, question_type_path=None, entities_path=None, speaker_mapping_path=None):
    examples = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')
    question_type = json.load(open(question_type_path, 'r', encoding='utf-8'))
    entities = pickle.load(open(entities_path, 'rb'), encoding='utf-8')
    speaker_mapping = json.load(open(speaker_mapping_path, 'r', encoding='utf-8'))
    tokenizer = ElectraTokenizerFast.from_pretrained('/mnt/sdb/ljn/pretrained/electra-large-discriminator')

    graph_related = []

    for example, feature in tqdm.tqdm(zip(examples, features)):
        speaker_counts = {}
        for utter_dict in example.utterances:
            speaker = ' & '.join(utter_dict['speakers'])
            if speaker not in speaker_counts.keys():
                speaker_counts[speaker] = 1
            else:
                speaker_counts[speaker] += 1
        speaker_id_map = {name: idx + 1 for idx, name in enumerate(speaker_counts.keys())}

        key_utterance = example.key_utterance

        conv_len = len(example.utterances)

        question = example.question

        question_type = list(set(question_type))

        six = ["what", "who", "where",
               "why", "how", "when",
               "does", "will", "is",
               "whare", "did", "was",
               "name", "whos"]

        question_words = question.split(' ')
        if question_words[0] in six:
            question_mask = [0] + [1] * len(tokenizer.encode(question_words[0], add_special_tokens=False)) + \
                            [0] * (31 - len(tokenizer.encode(question_words[0], add_special_tokens=False)))
        else:
            contained_q = []
            for qt in question_type:
                if qt in question:
                    contained_q.append(qt)
            max_qt_len = 0
            picked_q = ''
            for qt in contained_q:
                if len(qt.split(' ')) > max_qt_len:
                    picked_q = qt
                    max_qt_len = len(qt.split(' '))

            tokenized_q = tokenizer.encode(picked_q, add_special_tokens=False)

            question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
            if question_length > 32:
                while len(tokenizer.encode(question)) > 32:
                    question = question[:-1]
            remain_length_q = 32 - question_length
            question += ' '.join([tokenizer.pad_token] * remain_length_q)

            tokenized_question = tokenizer.encode(question)
            q_start_pos = match(tokenized_question, tokenized_q)
            if q_start_pos is None:
                question_mask = [0, 1] + [0] * 30
            else:
                question_mask = [0] * q_start_pos + [1] * len(tokenized_q) + [0] * (32 - q_start_pos - len(tokenized_q))

        qid = example.qid.split('-')[0]
        entity = entities[qid]
        speaker_list = []
        if entity is not None:
            for e in entity:
                if e.type == 'PERSON':
                    if e.text in speaker_mapping:
                        if len(speaker_mapping[e.text]) > 0:
                            contain_speaker = False
                            for spk in speaker_mapping[e.text]:
                                if type(spk) == list:
                                    for spk_in in spk:
                                        spk_in = spk_in.lower()
                                        if spk_in in speaker_id_map:
                                            contain_speaker = True
                                            break
                                    if contain_speaker:
                                        break
                                else:
                                    spk = spk.lower()
                                    if spk in speaker_id_map:
                                        contain_speaker = True
                                        break
                            if contain_speaker:
                                speaker_list.append(e.text)
        question_speaker_connection = {}
        question_speaker_mask = {}
        for spk in speaker_list:
            mapped_speakers = speaker_mapping[spk]
            if type(mapped_speakers[0]) == list:
                spk_in_list = spk.split()
                for sspk in spk_in_list:
                    if sspk in speaker_mapping:
                        mapped_speakers = speaker_mapping[sspk]
                        sspk = sspk.lower()
                        for mapped_speaker in mapped_speakers:
                            if type(mapped_speaker) == list:
                                mapped_speaker = mapped_speaker[0]
                            if mapped_speaker.lower() in speaker_id_map:
                                mapped_speaker = mapped_speaker.lower()
                                if sspk not in question_speaker_connection:
                                    tokenized_spk = tokenizer.encode(sspk, add_special_tokens=False)
                                    question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                                    if question_length > 32:
                                        while len(tokenizer.encode(question)) > 32:
                                            question = question[:-1]
                                    remain_length_q = 32 - question_length
                                    question += ' '.join([tokenizer.pad_token] * remain_length_q)
                                    tokenized_question = tokenizer.encode(question)

                                    spk_start_pos = match(tokenized_question, tokenized_spk)
                                    if spk_start_pos is not None:
                                        question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                        question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                                break
            else:
                spk = spk.lower()
                for mapped_speaker in mapped_speakers:
                    if mapped_speaker.lower() in speaker_id_map:
                        mapped_speaker = mapped_speaker.lower()
                        if spk not in question_speaker_connection:
                            tokenized_spk = tokenizer.encode(spk, add_special_tokens=False)
                            question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                            if question_length > 32:
                                while len(tokenizer.encode(question)) > 32:
                                    question = question[:-1]
                            remain_length_q = 32 - question_length
                            question += ' '.join([tokenizer.pad_token] * remain_length_q)
                            tokenized_question = tokenizer.encode(question)

                            spk_start_pos = match(tokenized_question, tokenized_spk)
                            if spk_start_pos is not None:
                                question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                        break

        # speaker_range_index = feature.speaker_range_index
        # speaker_position_mask = feature.speaker_position_mask
        input_ids = feature.input_ids
        utterances = example.utterances
        all_utterance_start_pos = []

        for i in range(len(input_ids)):
            if input_ids[i] == tokenizer.sep_token_id:
                if i == len(input_ids) - 1:
                    continue
                else:
                    all_utterance_start_pos.append(i+1)

        key_set = set()

        for k in key_utterance:
            max_k = min(k+3, conv_len)
            for i in range(k, max_k):
                key_set.add(i)
        key_set = sorted(list(key_set))

        word_node_list = []
        utter_mask = []
        speaker_node_map = {}
        speaker_list = []
        nodes_mapping = []
        nodes_mapping_mask = []
        num_node = 0
        num_words = 0
        n = 0
        for k in key_set:
            utter = utterances[k]
            num_node += 1
            k_speaker = ' & '.join(utter['speakers'])
            speaker_len = len(tokenizer.encode(k_speaker, add_special_tokens=False))
            s_start_pos = all_utterance_start_pos[k]
            u_start_pos = s_start_pos + speaker_len
            utter_len = len(tokenizer.encode(utter['utterance'], add_special_tokens=False)) + 1
            word_node_list.append(list(range(u_start_pos, u_start_pos+utter_len)))
            num_node += utter_len
            num_words += utter_len
            if speaker_id_map[k_speaker] not in speaker_node_map.keys():
                speaker_node_map[speaker_id_map[k_speaker]] = n
                n += 1
            speaker_list.append(speaker_node_map[speaker_id_map[k_speaker]])
            utter_mask.append([0]*u_start_pos + [1]*utter_len + [0]*(len(input_ids)-u_start_pos-utter_len))
            # speaker_node_list.append(speaker_id_map[k_speaker])
        num_node += len(speaker_node_map)
        num_node += (1 + len(question_speaker_mask))
        utter_start_ids = num_words + len(speaker_node_map)
        adj = torch.zeros(num_node, num_node, dtype=torch.long)
        accum_num_word = 0
        utter_ids = utter_start_ids
        for idx, k in enumerate(key_set):
            nodes_mapping += [0] * (all_utterance_start_pos[k] - len(nodes_mapping))
            nodes_mapping_mask += [0] * (all_utterance_start_pos[k] - len(nodes_mapping_mask))
            k_words = word_node_list[idx]
            k_speaker = speaker_list[idx]
            speaker_pos = num_words + k_speaker
            for k1, v1 in speaker_node_map.items():
                if k_speaker == v1:
                    for k2, v2 in speaker_id_map.items():
                        if k1 == v2:
                            speaker_name = k2
                            speaker_name = tokenizer.encode(speaker_name, add_special_tokens=False)
                            nodes_mapping += [speaker_pos] * len(speaker_name)
                            nodes_mapping_mask += [1] * len(speaker_name)
            k_num_word = len(k_words)
            nodes_mapping += list(range(accum_num_word, accum_num_word + k_num_word))
            nodes_mapping_mask += [1] * k_num_word

            for i in range(accum_num_word, accum_num_word + k_num_word):
                his_i = max(accum_num_word, i-n_gram)
                fut_i = min(i+n_gram+1, k_num_word+accum_num_word)
                for j in range(his_i, fut_i):
                    if i != j:
                        adj[i, j] = 1
                adj[i, speaker_pos] = 1
                adj[speaker_pos, i] = 1
                adj[i, utter_ids] = 1
                adj[utter_ids, i] = 1
            accum_num_word += k_num_word
            utter_ids += 1
        utter_ids = utter_start_ids

        if len(key_set) > 0:
            nodes_mapping = nodes_mapping + [0] * (len(input_ids) - len(nodes_mapping))
            nodes_mapping_mask = nodes_mapping_mask + [0] * (len(input_ids) - len(nodes_mapping_mask))

        for i in range(len(key_set)):
            his_i = max(0, i-context_range)
            fut_i = min(i+context_range+1, len(key_set))
            utter_i = key_set[i]
            adj[i+utter_ids, utter_ids+len(key_set)] = 1
            adj[utter_ids+len(key_set), i+utter_ids] = 1
            for his_ii in range(his_i, i):
                utter_his_i = key_set[his_ii]
                if his_ii != i:
                    if utter_i - utter_his_i <= context_range:
                        adj[i + utter_ids, his_ii + utter_ids] = 1
            for fut_ii in range(i, fut_i):
                utter_fut_i = key_set[fut_ii]
                if fut_ii != i:
                    if utter_fut_i - utter_i <= context_range:
                        adj[i + utter_ids, fut_ii + utter_ids] = 1
        if len(question_speaker_mask) > 0:
            i_count = 0
            question_speaker_start = utter_ids + len(key_set) + 1
            for _, s_c in question_speaker_connection.items():
                question_spk = question_speaker_start + i_count
                adj[question_spk, question_speaker_start-1] = 1
                adj[question_speaker_start-1, question_spk] = 1
                for j_count in range(len(question_speaker_connection)):
                    if question_spk != j_count+question_speaker_start:
                        adj[question_spk, j_count+question_speaker_start] = 1
                        adj[j_count+question_speaker_start, question_spk] = 1
                if s_c in speaker_node_map:
                    s_c_id = speaker_node_map[s_c] + num_words
                    adj[s_c_id, question_spk] = 1
                    adj[question_spk, s_c_id] = 1
                i_count += 1
        related_speaker = list(speaker_node_map.keys())

        word_nodes = []
        for w in word_node_list:
            word_nodes += w

        graph_related.append({'word_nodes': word_nodes, 'utter_mask': utter_mask,
                              'related_speaker': related_speaker, 'adj': adj,
                              'nodes_mapping': nodes_mapping,
                              'nodes_mapping_mask': nodes_mapping_mask,
                              'num_words': num_words,
                              'num_utter': len(key_set),
                              'num_speaker': len(speaker_node_map),
                              'num_question_speaker': len(question_speaker_connection),
                              'question_node_mask': question_mask,
                              'question_speaker_mask': question_speaker_mask})
    pickle.dump(graph_related, open(save_path, 'wb'))


def get_data_1(data_path, feature_path, save_path, n_gram=1, context_range=2, question_type_path=None, entities_path=None, speaker_mapping_path=None):
    examples = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')
    question_type = json.load(open(question_type_path, 'r', encoding='utf-8'))
    entities = pickle.load(open(entities_path, 'rb'), encoding='utf-8')
    speaker_mapping = json.load(open(speaker_mapping_path, 'r', encoding='utf-8'))
    tokenizer = ElectraTokenizerFast.from_pretrained('/mnt/sdb/ljn/pretrained/electra-large-discriminator')
    nlp_pipeline = stanza.Pipeline('en')

    graph_related = []

    for example, feature in tqdm.tqdm(zip(examples, features)):
        speaker_counts = {}
        for utter_dict in example.utterances:
            speaker = ' & '.join(utter_dict['speakers'])
            if speaker != '#note#':
                if speaker not in speaker_counts.keys():
                    speaker_counts[speaker] = 1
                else:
                    speaker_counts[speaker] += 1
        speaker_id_map = {name: idx + 1 for idx, name in enumerate(speaker_counts.keys())}

        speaker_position_mask = [[0] * 32 for i in range(len(speaker_id_map.keys()))]
        for speaker in speaker_id_map.keys():
            speaker_position = [1] * len(tokenizer.encode(speaker, add_special_tokens=False))
            for utterance in example.utterances:
                speaker_ = ' & '.join(utterance['speakers'])
                utter_text = [0] * (len(tokenizer.encode(utterance['utterance'], add_special_tokens=False)) + 2)
                if speaker_ == speaker:
                    speaker_position_mask[speaker_id_map[speaker] - 1] += speaker_position
                else:
                    speaker_po = [0] * len(tokenizer.encode(speaker_, add_special_tokens=False))
                    speaker_position_mask[speaker_id_map[speaker] - 1] += speaker_po
                speaker_position_mask[speaker_id_map[speaker] - 1] += utter_text
            speaker_position_mask[speaker_id_map[speaker] - 1] += [0] * (512 - len(speaker_position_mask[speaker_id_map[speaker] - 1]))
        speaker_position_mask = [[0] * 512] + speaker_position_mask

        key_utterance = example.key_utterance

        conv_len = len(example.utterances)

        question = example.question

        question_type = list(set(question_type))

        six = ["what", "who", "where",
               "why", "how", "when",
               "does", "will", "is",
               "whare", "did", "was",
               "name", "whos"]

        question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
        if question_length > 32:
            while len(tokenizer.encode(question)) > 32:
                question = question[:-1]
        # question_length = len(tokenizer.encode(question))
        remain_length_q = 32 - question_length
        question += ' '.join([tokenizer.pad_token] * remain_length_q)
        # question_mask = [0] + [1] * (question_length - 2) + [0] * (33 - question_length)

        question_words = question.split(' ')
        if question_words[0] in six:
            question_mask = [0] + [1] * len(tokenizer.encode(question_words[0], add_special_tokens=False)) + \
                            [0] * (31 - len(tokenizer.encode(question_words[0], add_special_tokens=False)))
        else:
            contained_q = []
            for qt in question_type:
                if qt in question:
                    contained_q.append(qt)
            max_qt_len = 0
            picked_q = ''
            for qt in contained_q:
                if len(qt.split(' ')) > max_qt_len:
                    picked_q = qt
                    max_qt_len = len(qt.split(' '))

            tokenized_q = tokenizer.encode(picked_q, add_special_tokens=False)

            # question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
            # if question_length > 32:
            #     while len(tokenizer.encode(question)) > 32:
            #         question = question[:-1]
            # remain_length_q = 32 - question_length
            # question += ' '.join([tokenizer.pad_token] * remain_length_q)
            # question_mask = [0] + [1] * (question_length - 2) + [0] * (33 - question_length)

            tokenized_question = tokenizer.encode(question)
            q_start_pos = match(tokenized_question, tokenized_q)
            if q_start_pos is None:
                question_mask = [0, 1] + [0] * 30
            else:
                question_mask = [0] * q_start_pos + [1] * len(tokenized_q) + [0] * (32 - q_start_pos - len(tokenized_q))

        qid = example.qid.split('-')[0]
        entity = entities[qid]
        speaker_list = []
        if entity is not None:
            for e in entity:
                if e.type == 'PERSON':
                    if e.text in speaker_mapping:
                        if len(speaker_mapping[e.text]) > 0:
                            contain_speaker = False
                            for spk in speaker_mapping[e.text]:
                                if type(spk) == list:
                                    for spk_in in spk:
                                        spk_in = spk_in.lower()
                                        if spk_in in speaker_id_map:
                                            contain_speaker = True
                                            break
                                    if contain_speaker:
                                        break
                                else:
                                    spk = spk.lower()
                                    if spk in speaker_id_map:
                                        contain_speaker = True
                                        break
                            if contain_speaker:
                                speaker_list.append(e.text)
        question_speaker_connection = {}
        question_speaker_mask = {}
        for spk in speaker_list:
            mapped_speakers = speaker_mapping[spk]
            if type(mapped_speakers[0]) == list:
                spk_in_list = spk.split()
                for sspk in spk_in_list:
                    if sspk in speaker_mapping:
                        mapped_speakers = speaker_mapping[sspk]
                        sspk = sspk.lower()
                        for mapped_speaker in mapped_speakers:
                            if type(mapped_speaker) == list:
                                mapped_speaker = mapped_speaker[0]
                            if mapped_speaker.lower() in speaker_id_map:
                                mapped_speaker = mapped_speaker.lower()
                                if sspk not in question_speaker_connection:
                                    tokenized_spk = tokenizer.encode(sspk, add_special_tokens=False)
                                    question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                                    if question_length > 32:
                                        while len(tokenizer.encode(question)) > 32:
                                            question = question[:-1]
                                    remain_length_q = 32 - question_length
                                    question += ' '.join([tokenizer.pad_token] * remain_length_q)
                                    tokenized_question = tokenizer.encode(question)

                                    spk_start_pos = match(tokenized_question, tokenized_spk)
                                    if spk_start_pos is not None:
                                        question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                        question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                                break
            else:
                spk = spk.lower()
                for mapped_speaker in mapped_speakers:
                    if mapped_speaker.lower() in speaker_id_map:
                        mapped_speaker = mapped_speaker.lower()
                        if spk not in question_speaker_connection:
                            tokenized_spk = tokenizer.encode(spk, add_special_tokens=False)
                            question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                            if question_length > 32:
                                while len(tokenizer.encode(question)) > 32:
                                    question = question[:-1]
                            remain_length_q = 32 - question_length
                            question += ' '.join([tokenizer.pad_token] * remain_length_q)
                            tokenized_question = tokenizer.encode(question)

                            spk_start_pos = match(tokenized_question, tokenized_spk)
                            if spk_start_pos is not None:
                                question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                        break

        # speaker_range_index = feature.speaker_range_index
        # speaker_position_mask = feature.speaker_position_mask
        input_ids = feature.input_ids
        utterances = example.utterances
        utterances_cased = example.utterances_cased
        all_utterance_start_pos = []

        for i in range(len(input_ids)):
            if input_ids[i] == tokenizer.sep_token_id:
                if i == len(input_ids) - 1:
                    continue
                else:
                    all_utterance_start_pos.append(i+1)

        key_set = set()

        for k in key_utterance:
            max_k = min(k+3, conv_len)
            for i in range(k, max_k):
                key_set.add(i)
        key_set = sorted(list(key_set))

        word_node_list = []
        utter_mask = []
        speaker_node_map = {}
        speaker_list = []
        nodes_mapping = []
        nodes_mapping_mask = []
        num_node = 0
        num_words = 0
        n = 0
        for k in key_set:
            utter = utterances[k]
            num_node += 1
            k_speaker = ' & '.join(utter['speakers'])
            speaker_len = len(tokenizer.encode(k_speaker, add_special_tokens=False))
            s_start_pos = all_utterance_start_pos[k]
            u_start_pos = s_start_pos + speaker_len
            utter_len = len(tokenizer.encode(utter['utterance'], add_special_tokens=False)) + 1
            word_node_list.append(list(range(u_start_pos, u_start_pos+utter_len)))
            num_node += utter_len
            num_words += utter_len
            if k_speaker != '#note#':
                if speaker_id_map[k_speaker] not in speaker_node_map.keys():
                    speaker_node_map[speaker_id_map[k_speaker]] = n
                    n += 1
                speaker_list.append(speaker_node_map[speaker_id_map[k_speaker]])
            else:
                speaker_list.append([])
            utter_mask.append([0] * s_start_pos + [1] * speaker_len + [0] * (len(input_ids) - s_start_pos - speaker_len))
            # speaker_node_list.append(speaker_id_map[k_speaker])
        find_scene = 0
        for k in key_set:
            utter = utterances[k]
            utter_cased = utterances_cased[k]
            k_speaker = ' & '.join(utter['speakers'])
            if k_speaker == '#note#':
                utterance_cased = utter_cased['utterance']
                doc = nlp_pipeline(utterance_cased)
                if len(doc.sentences[0].ents) > 0:
                    for ent in doc.sentences[0].ents:
                        if ent.type == 'PERSON':
                            sss = ent.text
                            if sss in speaker_mapping:
                                sss_mapped = speaker_mapping[sss]
                                for s_mp in sss_mapped:
                                    s_mp = s_mp.lower()
                                    if s_mp in speaker_id_map:
                                        if speaker_id_map[s_mp] in speaker_node_map:
                                            speaker_list[find_scene].append([speaker_node_map[speaker_id_map[s_mp]], sss.lower()])
                                            # speaker_list[find_scene].append(speaker_node_map[speaker_id_map[s_mp]])
                                            break
            find_scene += 1
        num_node += len(speaker_node_map)
        num_node += (1 + len(question_speaker_mask))
        utter_start_ids = num_words + len(speaker_node_map)
        question_node_ids = num_words + len(speaker_node_map) + len(utter_mask)
        # question_node_ids = num_words + len(speaker_node_map)
        adj = torch.zeros(num_node, num_node, dtype=torch.long)
        rel_adj = torch.zeros(num_node, num_node, dtype=torch.long)
        rel_adj[question_node_ids, question_node_ids] = 3    # self-connected
        node_type = [1] * num_words + [2] * len(speaker_node_map) + [3] * len(utter_mask) + [4] + [5] * len(question_speaker_connection)
        accum_num_word = 0
        utter_ids = utter_start_ids
        for idx, k in enumerate(key_set):
            nodes_mapping += [0] * (all_utterance_start_pos[k] - len(nodes_mapping))
            nodes_mapping_mask += [0] * (all_utterance_start_pos[k] - len(nodes_mapping_mask))
            k_words = word_node_list[idx]
            k_speaker = speaker_list[idx]
            if type(k_speaker) == list:
                speaker_name = '#note#'
                speaker_name = tokenizer.encode(speaker_name, add_special_tokens=False)
                nodes_mapping += [0] * len(speaker_name)
                nodes_mapping_mask += [1] * len(speaker_name)
                # speaker_pos = [k_s + num_words for k_s in k_speaker]
                speaker_pos = [[k_s[0] + num_words, k_s[1]] for k_s in k_speaker]
            else:
                speaker_pos = num_words + k_speaker
                for k1, v1 in speaker_node_map.items():
                    if k_speaker == v1:
                        for k2, v2 in speaker_id_map.items():
                            if k1 == v2:
                                speaker_name = k2
                                speaker_name = tokenizer.encode(speaker_name, add_special_tokens=False)
                                nodes_mapping += [speaker_pos] * len(speaker_name)
                                nodes_mapping_mask += [1] * len(speaker_name)
            k_num_word = len(k_words)
            nodes_mapping += list(range(accum_num_word, accum_num_word + k_num_word))
            nodes_mapping_mask += [1] * k_num_word

            # if type(speaker_pos) == list:
            #     utter = tokenizer.encode(' : ' + utterances[k]['utterance'], add_special_tokens=False)
            #     speaker_count_pos = {}
            #     for s_p in speaker_pos:
            #         if s_p[1] not in speaker_count_pos:
            #             speaker_count_pos[s_p[1]] = 0
            #     for s_p in speaker_pos:
            #         s_p_idx = s_p[0]
            #         s_p_begin_idx = speaker_count_pos[s_p[1]]
            #         s_p_name = tokenizer.encode(s_p[1], add_special_tokens=False)
            #         s_begin = match(utter, s_p_name, s_p_begin_idx)
            #         for i in range(len(s_p_name)):
            #             adj[accum_num_word+s_begin+i, s_p_idx] = 1
            #             adj[s_p_idx, accum_num_word+s_begin+i] = 1
            #         speaker_count_pos[s_p[1]] = s_begin + len(s_p_name)

            for i in range(accum_num_word, accum_num_word + k_num_word):
                his_i = max(accum_num_word, i-n_gram)
                fut_i = min(i+n_gram+1, k_num_word+accum_num_word)
                for j in range(his_i, fut_i):
                    if i != j:
                        adj[i, j] = 1
                    rel_adj[i, j] = 1    # self-connected
                if type(speaker_pos) == list:
                    # pass
                    for s_p in speaker_pos:
                        adj[i, s_p[0]] = 1
                        adj[s_p[0], i] = 1

                        rel_adj[i, s_p[0]] = 1
                        rel_adj[s_p[0], i] = 1
                else:
                    rel_adj[speaker_pos, speaker_pos] = 1  # self-connected

                    adj[i, speaker_pos] = 1
                    adj[speaker_pos, i] = 1

                    rel_adj[i, speaker_pos] = 1
                    rel_adj[speaker_pos, i] = 1
                adj[i, utter_ids] = 1
                adj[utter_ids, i] = 1
                rel_adj[i, utter_ids] = 1
                rel_adj[utter_ids, i] = 1

                adj[i, question_node_ids] = 1
                adj[question_node_ids, i] = 1
                rel_adj[i, question_node_ids] = 2
                rel_adj[question_node_ids, i] = 2
            accum_num_word += k_num_word
            utter_ids += 1
        utter_ids = utter_start_ids

        if len(key_set) > 0:
            nodes_mapping = nodes_mapping + [0] * (len(input_ids) - len(nodes_mapping))
            nodes_mapping_mask = nodes_mapping_mask + [0] * (len(input_ids) - len(nodes_mapping_mask))

        for i in range(len(key_set)):
            his_i = max(0, i-context_range)
            fut_i = min(i+context_range+1, len(key_set))
            utter_i = key_set[i]
            adj[i+utter_ids, utter_ids+len(key_set)] = 1
            adj[utter_ids+len(key_set), i+utter_ids] = 1
            rel_adj[i+utter_ids, utter_ids+len(key_set)] = 4
            rel_adj[utter_ids+len(key_set), i+utter_ids] = 4
            for his_ii in range(his_i, i):
                utter_his_i = key_set[his_ii]
                if his_ii != i:
                    if utter_i - utter_his_i <= context_range:
                        adj[i + utter_ids, his_ii + utter_ids] = 1
                        rel_adj[i+utter_ids, his_ii+utter_ids] = 1
            rel_adj[i+utter_ids, i+utter_ids] = 1
            for fut_ii in range(i, fut_i):
                utter_fut_i = key_set[fut_ii]
                if fut_ii != i:
                    if utter_fut_i - utter_i <= context_range:
                        adj[i + utter_ids, fut_ii + utter_ids] = 1
                        rel_adj[i + utter_ids, fut_ii + utter_ids] = 1

        if len(question_speaker_mask) > 0:
            i_count = 0
            # question_speaker_start = utter_ids + len(key_set) + 1
            question_speaker_start = question_node_ids + 1
            for _, s_c in question_speaker_connection.items():
                question_spk = question_speaker_start + i_count
                adj[question_spk, question_speaker_start-1] = 1
                adj[question_speaker_start-1, question_spk] = 1
                rel_adj[question_spk, question_speaker_start-1] = 3
                rel_adj[question_speaker_start-1, question_spk] = 3
                rel_adj[question_spk, question_spk] = 3    # self-connected
                for j_count in range(len(question_speaker_connection)):
                    if question_spk != j_count+question_speaker_start:
                        adj[question_spk, j_count+question_speaker_start] = 1
                        adj[j_count+question_speaker_start, question_spk] = 1
                    rel_adj[question_spk, j_count+question_speaker_start] = 3
                    rel_adj[j_count+question_speaker_start, question_spk] = 3
                if s_c in speaker_node_map:
                    s_c_id = speaker_node_map[s_c] + num_words
                    adj[s_c_id, question_spk] = 1
                    adj[question_spk, s_c_id] = 1
                    rel_adj[s_c_id, question_spk] = 5
                    rel_adj[question_spk, s_c_id] = 5
                i_count += 1
        related_speaker = list(speaker_node_map.keys())

        word_nodes = []
        for w in word_node_list:
            word_nodes += w

        graph_related.append({'word_nodes': word_nodes,
                              'related_speaker': related_speaker,
                              'adj': adj,
                              'rel_adj': rel_adj,
                              'nodes_mapping': nodes_mapping,
                              'nodes_mapping_mask': nodes_mapping_mask,
                              'utter_mask': utter_mask,
                              'node_type': node_type,
                              'num_words': num_words,
                              'num_utter': len(key_set),
                              'num_speaker': len(speaker_node_map),
                              'num_question_speaker': len(question_speaker_connection),
                              'question_node_mask': question_mask,
                              'question_speaker_mask': question_speaker_mask,
                              'speaker_position_mask': speaker_position_mask})
    pickle.dump(graph_related, open(save_path, 'wb'))


def get_data_2(data_path, feature_path, save_path, n_gram=1, context_range=2, question_type_path=None, entities_path=None, speaker_mapping_path=None):
    examples = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')
    question_type = json.load(open(question_type_path, 'r', encoding='utf-8'))
    entities = pickle.load(open(entities_path, 'rb'), encoding='utf-8')
    speaker_mapping = json.load(open(speaker_mapping_path, 'r', encoding='utf-8'))
    tokenizer = ElectraTokenizerFast.from_pretrained('/mnt/sdb/ljn/pretrained/electra-large-discriminator')
    nlp_pipeline = stanza.Pipeline('en')

    graph_related = []

    for example, feature in tqdm.tqdm(zip(examples, features)):
        speaker_counts = {}
        for utter_dict in example.utterances:
            speaker = ' & '.join(utter_dict['speakers'])
            if speaker != '#note#':
                if speaker not in speaker_counts.keys():
                    speaker_counts[speaker] = 1
                else:
                    speaker_counts[speaker] += 1
        speaker_id_map = {name: idx + 1 for idx, name in enumerate(speaker_counts.keys())}

        speaker_position_mask = [[0] * 32 for i in range(len(speaker_id_map.keys()))]
        for speaker in speaker_id_map.keys():
            speaker_position = [1] * len(tokenizer.encode(speaker, add_special_tokens=False))
            for utterance in example.utterances:
                speaker_ = ' & '.join(utterance['speakers'])
                utter_text = [0] * (len(tokenizer.encode(utterance['utterance'], add_special_tokens=False)) + 2)
                if speaker_ == speaker:
                    speaker_position_mask[speaker_id_map[speaker] - 1] += speaker_position
                else:
                    speaker_po = [0] * len(tokenizer.encode(speaker_, add_special_tokens=False))
                    speaker_position_mask[speaker_id_map[speaker] - 1] += speaker_po
                speaker_position_mask[speaker_id_map[speaker] - 1] += utter_text
            speaker_position_mask[speaker_id_map[speaker] - 1] += [0] * (512 - len(speaker_position_mask[speaker_id_map[speaker] - 1]))
        speaker_position_mask = [[0] * 512] + speaker_position_mask

        key_utterance = example.key_utterance

        conv_len = len(example.utterances)

        question = example.question

        question_type = list(set(question_type))

        six = ["what", "who", "where",
               "why", "how", "when",
               "does", "will", "is",
               "whare", "did", "was",
               "name", "whos"]

        question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
        if question_length > 32:
            while len(tokenizer.encode(question)) > 32:
                question = question[:-1]
        # question_length = len(tokenizer.encode(question))
        remain_length_q = 32 - question_length
        question += ' '.join([tokenizer.pad_token] * remain_length_q)
        # question_mask = [0] + [1] * (question_length - 2) + [0] * (33 - question_length)

        question_words = question.split(' ')
        if question_words[0] in six:
            question_mask = [0] + [1] * len(tokenizer.encode(question_words[0], add_special_tokens=False)) + \
                            [0] * (31 - len(tokenizer.encode(question_words[0], add_special_tokens=False)))
        else:
            contained_q = []
            for qt in question_type:
                if qt in question:
                    contained_q.append(qt)
            max_qt_len = 0
            picked_q = ''
            for qt in contained_q:
                if len(qt.split(' ')) > max_qt_len:
                    picked_q = qt
                    max_qt_len = len(qt.split(' '))

            tokenized_q = tokenizer.encode(picked_q, add_special_tokens=False)

            # question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
            # if question_length > 32:
            #     while len(tokenizer.encode(question)) > 32:
            #         question = question[:-1]
            # remain_length_q = 32 - question_length
            # question += ' '.join([tokenizer.pad_token] * remain_length_q)
            # question_mask = [0] + [1] * (question_length - 2) + [0] * (33 - question_length)

            tokenized_question = tokenizer.encode(question)
            q_start_pos = match(tokenized_question, tokenized_q)
            if q_start_pos is None:
                question_mask = [0, 1] + [0] * 30
            else:
                question_mask = [0] * q_start_pos + [1] * len(tokenized_q) + [0] * (32 - q_start_pos - len(tokenized_q))

        qid = example.qid.split('-')[0]
        entity = entities[qid]
        speaker_list = []
        if entity is not None:
            for e in entity:
                if e.type == 'PERSON':
                    if e.text in speaker_mapping:
                        if len(speaker_mapping[e.text]) > 0:
                            contain_speaker = False
                            for spk in speaker_mapping[e.text]:
                                if type(spk) == list:
                                    for spk_in in spk:
                                        spk_in = spk_in.lower()
                                        if spk_in in speaker_id_map:
                                            contain_speaker = True
                                            break
                                    if contain_speaker:
                                        break
                                else:
                                    spk = spk.lower()
                                    if spk in speaker_id_map:
                                        contain_speaker = True
                                        break
                            if contain_speaker:
                                speaker_list.append(e.text)
        question_speaker_connection = {}
        question_speaker_mask = {}
        for spk in speaker_list:
            mapped_speakers = speaker_mapping[spk]
            if type(mapped_speakers[0]) == list:
                spk_in_list = spk.split()
                for sspk in spk_in_list:
                    if sspk in speaker_mapping:
                        mapped_speakers = speaker_mapping[sspk]
                        sspk = sspk.lower()
                        for mapped_speaker in mapped_speakers:
                            if type(mapped_speaker) == list:
                                mapped_speaker = mapped_speaker[0]
                            if mapped_speaker.lower() in speaker_id_map:
                                mapped_speaker = mapped_speaker.lower()
                                if sspk not in question_speaker_connection:
                                    tokenized_spk = tokenizer.encode(sspk, add_special_tokens=False)
                                    question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                                    if question_length > 32:
                                        while len(tokenizer.encode(question)) > 32:
                                            question = question[:-1]
                                    remain_length_q = 32 - question_length
                                    question += ' '.join([tokenizer.pad_token] * remain_length_q)
                                    tokenized_question = tokenizer.encode(question)

                                    spk_start_pos = match(tokenized_question, tokenized_spk)
                                    if spk_start_pos is not None:
                                        question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                        question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                                break
            else:
                spk = spk.lower()
                for mapped_speaker in mapped_speakers:
                    if mapped_speaker.lower() in speaker_id_map:
                        mapped_speaker = mapped_speaker.lower()
                        if spk not in question_speaker_connection:
                            tokenized_spk = tokenizer.encode(spk, add_special_tokens=False)
                            question_length = len(tokenizer.encode(question))  # including [CLS] and [SEP]
                            if question_length > 32:
                                while len(tokenizer.encode(question)) > 32:
                                    question = question[:-1]
                            remain_length_q = 32 - question_length
                            question += ' '.join([tokenizer.pad_token] * remain_length_q)
                            tokenized_question = tokenizer.encode(question)

                            spk_start_pos = match(tokenized_question, tokenized_spk)
                            if spk_start_pos is not None:
                                question_speaker_connection[spk] = speaker_id_map[mapped_speaker]
                                question_speaker_mask[spk] = [0] * spk_start_pos + [1] * len(tokenized_spk) + [0] * (32 - spk_start_pos - len(tokenized_spk))
                        break

        # speaker_range_index = feature.speaker_range_index
        # speaker_position_mask = feature.speaker_position_mask
        input_ids = feature.input_ids
        utterances = example.utterances
        utterances_cased = example.utterances_cased
        all_utterance_start_pos = []

        for i in range(len(input_ids)):
            if input_ids[i] == tokenizer.sep_token_id:
                if i == len(input_ids) - 1:
                    continue
                else:
                    all_utterance_start_pos.append(i+1)

        # key_set = list(range(len(utterances)))

        key_set = set()

        for k in key_utterance:
            max_k = min(k + 3, conv_len)
            for i in range(k, max_k):
                key_set.add(i)
        key_set = sorted(list(key_set))

        word_node_list = []
        utter_mask = []
        utter_spk_mask = []
        speaker_node_map = {}
        speaker_list = []
        nodes_mapping = []
        nodes_mapping_mask = []
        num_node = 0
        num_words = 0
        n = 0
        for k in key_set:
            utter = utterances[k]
            num_node += 1
            k_speaker = ' & '.join(utter['speakers'])
            speaker_len = len(tokenizer.encode(k_speaker, add_special_tokens=False))
            s_start_pos = all_utterance_start_pos[k]
            u_start_pos = s_start_pos + speaker_len
            utter_len = len(tokenizer.encode(utter['utterance'], add_special_tokens=False)) + 1
            word_node_list.append(list(range(u_start_pos, u_start_pos+utter_len)))
            num_node += utter_len
            num_words += utter_len
            if k_speaker != '#note#':
                if speaker_id_map[k_speaker] not in speaker_node_map.keys():
                    speaker_node_map[speaker_id_map[k_speaker]] = n
                    n += 1
                speaker_list.append(speaker_node_map[speaker_id_map[k_speaker]])
            else:
                speaker_list.append([])
            utter_spk_mask.append([0] * s_start_pos + [1] * speaker_len + [0] * (len(input_ids) - s_start_pos - speaker_len))
            utter_mask.append([0] * u_start_pos + [1] * utter_len + [0] * (len(input_ids) - u_start_pos - utter_len))
            # speaker_node_list.append(speaker_id_map[k_speaker])
        find_scene = 0
        for k in key_set:
            utter = utterances[k]
            utter_cased = utterances_cased[k]
            k_speaker = ' & '.join(utter['speakers'])
            if k_speaker == '#note#':
                utterance_cased = utter_cased['utterance']
                doc = nlp_pipeline(utterance_cased)
                if len(doc.sentences[0].ents) > 0:
                    for ent in doc.sentences[0].ents:
                        if ent.type == 'PERSON':
                            sss = ent.text
                            if sss in speaker_mapping:
                                sss_mapped = speaker_mapping[sss]
                                for s_mp in sss_mapped:
                                    s_mp = s_mp.lower()
                                    if s_mp in speaker_id_map:
                                        if speaker_id_map[s_mp] in speaker_node_map:
                                            speaker_list[find_scene].append([speaker_node_map[speaker_id_map[s_mp]], sss.lower()])
                                            # speaker_list[find_scene].append(speaker_node_map[speaker_id_map[s_mp]])
                                            break
            find_scene += 1
        num_node += len(speaker_node_map)
        num_node += (1 + len(question_speaker_mask))
        utter_start_ids = num_words + len(speaker_node_map)
        question_node_ids = num_words + len(speaker_node_map) + len(utter_mask)
        # question_node_ids = num_words + len(speaker_node_map)
        adj = torch.zeros(num_node, num_node, dtype=torch.long)
        # rel_adj = torch.zeros(num_node, num_node, dtype=torch.long)
        # rel_adj[question_node_ids, question_node_ids] = 3    # self-connected
        node_type = [1] * num_words + [2] * len(speaker_node_map) + [3] * len(utter_mask) + [4] + [5] * len(question_speaker_connection)
        node_type_no_utter = [1] * num_words + [2] * len(speaker_node_map) + [3] + [4] * len(question_speaker_connection)
        node_scene_type = [1] * num_words + [2] * len(speaker_node_map)
        accum_num_word = 0
        utter_ids = utter_start_ids
        for idx, k in enumerate(key_set):
            nodes_mapping += [0] * (all_utterance_start_pos[k] - len(nodes_mapping))
            nodes_mapping_mask += [0] * (all_utterance_start_pos[k] - len(nodes_mapping_mask))
            k_words = word_node_list[idx]
            k_speaker = speaker_list[idx]
            if type(k_speaker) == list:
                node_scene_type = node_scene_type + [3]
                speaker_name = '#note#'
                speaker_name = tokenizer.encode(speaker_name, add_special_tokens=False)
                nodes_mapping += [0] * len(speaker_name)
                nodes_mapping_mask += [1] * len(speaker_name)
                # speaker_pos = [k_s + num_words for k_s in k_speaker]
                speaker_pos = [[k_s[0] + num_words, k_s[1]] for k_s in k_speaker]
            else:
                node_scene_type = node_scene_type + [4]
                speaker_pos = num_words + k_speaker
                for k1, v1 in speaker_node_map.items():
                    if k_speaker == v1:
                        for k2, v2 in speaker_id_map.items():
                            if k1 == v2:
                                speaker_name = k2
                                speaker_name = tokenizer.encode(speaker_name, add_special_tokens=False)
                                nodes_mapping += [speaker_pos] * len(speaker_name)
                                nodes_mapping_mask += [1] * len(speaker_name)
            k_num_word = len(k_words)
            nodes_mapping += list(range(accum_num_word, accum_num_word + k_num_word))
            nodes_mapping_mask += [1] * k_num_word

            # if type(speaker_pos) == list:
            #     utter = tokenizer.encode(' : ' + utterances[k]['utterance'], add_special_tokens=False)
            #     speaker_count_pos = {}
            #     for s_p in speaker_pos:
            #         if s_p[1] not in speaker_count_pos:
            #             speaker_count_pos[s_p[1]] = 0
            #     for s_p in speaker_pos:
            #         s_p_idx = s_p[0]
            #         s_p_begin_idx = speaker_count_pos[s_p[1]]
            #         s_p_name = tokenizer.encode(s_p[1], add_special_tokens=False)
            #         s_begin = match(utter, s_p_name, s_p_begin_idx)
            #         for i in range(len(s_p_name)):
            #             adj[accum_num_word+s_begin+i, s_p_idx] = 1
            #             adj[s_p_idx, accum_num_word+s_begin+i] = 1
            #         speaker_count_pos[s_p[1]] = s_begin + len(s_p_name)

            for i in range(accum_num_word, accum_num_word + k_num_word):
                his_i = max(accum_num_word, i-n_gram)
                fut_i = min(i+n_gram+1, k_num_word+accum_num_word)
                for j in range(his_i, fut_i):
                    if i != j:
                        adj[i, j] = 1
                    # rel_adj[i, j] = 1    # self-connected
                if type(speaker_pos) == list:
                    # pass
                    for s_p in speaker_pos:
                        adj[i, s_p[0]] = 1
                        adj[s_p[0], i] = 1

                        # rel_adj[i, s_p[0]] = 1
                        # rel_adj[s_p[0], i] = 1
                else:
                    # rel_adj[speaker_pos, speaker_pos] = 1  # self-connected

                    adj[i, speaker_pos] = 1
                    adj[speaker_pos, i] = 1

                    # rel_adj[i, speaker_pos] = 1
                    # rel_adj[speaker_pos, i] = 1
                adj[i, utter_ids] = 1
                adj[utter_ids, i] = 1
                # rel_adj[i, utter_ids] = 1
                # rel_adj[utter_ids, i] = 1

                adj[i, question_node_ids] = 1
                adj[question_node_ids, i] = 1
                # rel_adj[i, question_node_ids] = 2
                # rel_adj[question_node_ids, i] = 2
            accum_num_word += k_num_word
            utter_ids += 1

        node_scene_type = node_scene_type + [5] + [6] * len(question_speaker_connection)

        utter_ids = utter_start_ids

        if len(key_set) > 0:
            nodes_mapping = nodes_mapping + [0] * (len(input_ids) - len(nodes_mapping))
            nodes_mapping_mask = nodes_mapping_mask + [0] * (len(input_ids) - len(nodes_mapping_mask))

        for i in range(len(key_set)):
            his_i = max(0, i-context_range)
            fut_i = min(i+context_range+1, len(key_set))
            utter_i = key_set[i]
            adj[i+utter_ids, utter_ids+len(key_set)] = 1
            adj[utter_ids+len(key_set), i+utter_ids] = 1
            # rel_adj[i+utter_ids, utter_ids+len(key_set)] = 4
            # rel_adj[utter_ids+len(key_set), i+utter_ids] = 4
            for his_ii in range(his_i, i):
                utter_his_i = key_set[his_ii]
                if his_ii != i:
                    if utter_i - utter_his_i <= context_range:
                        adj[i + utter_ids, his_ii + utter_ids] = 1
                        # rel_adj[i+utter_ids, his_ii+utter_ids] = 1
            # rel_adj[i+utter_ids, i+utter_ids] = 1
            for fut_ii in range(i, fut_i):
                utter_fut_i = key_set[fut_ii]
                if fut_ii != i:
                    if utter_fut_i - utter_i <= context_range:
                        adj[i + utter_ids, fut_ii + utter_ids] = 1
                        # rel_adj[i + utter_ids, fut_ii + utter_ids] = 1

        if len(question_speaker_mask) > 0:
            i_count = 0
            # question_speaker_start = utter_ids + len(key_set) + 1
            question_speaker_start = question_node_ids + 1
            for _, s_c in question_speaker_connection.items():
                question_spk = question_speaker_start + i_count
                adj[question_spk, question_speaker_start-1] = 1
                adj[question_speaker_start-1, question_spk] = 1
                # rel_adj[question_spk, question_speaker_start-1] = 3
                # rel_adj[question_speaker_start-1, question_spk] = 3
                # rel_adj[question_spk, question_spk] = 3    # self-connected
                for j_count in range(len(question_speaker_connection)):
                    if question_spk != j_count+question_speaker_start:
                        adj[question_spk, j_count+question_speaker_start] = 1
                        adj[j_count+question_speaker_start, question_spk] = 1
                    # rel_adj[question_spk, j_count+question_speaker_start] = 3
                    # rel_adj[j_count+question_speaker_start, question_spk] = 3
                if s_c in speaker_node_map:
                    s_c_id = speaker_node_map[s_c] + num_words
                    adj[s_c_id, question_spk] = 1
                    adj[question_spk, s_c_id] = 1
                    # rel_adj[s_c_id, question_spk] = 5
                    # rel_adj[question_spk, s_c_id] = 5
                i_count += 1
        related_speaker = list(speaker_node_map.keys())

        word_nodes = []
        for w in word_node_list:
            word_nodes += w

        graph_related.append({'word_nodes': word_nodes,
                              'related_speaker': related_speaker,
                              'adj': adj.numpy(),
                              'nodes_mapping': nodes_mapping,
                              'nodes_mapping_mask': nodes_mapping_mask,
                              'utter_mask': utter_mask,
                              'utter_spk_mask': utter_spk_mask,
                              'node_type': node_type,
                              'node_scene_type': node_scene_type,
                              'node_type_no_utter': node_type_no_utter,
                              'num_words': num_words,
                              'num_utter': len(key_set),
                              'num_speaker': len(speaker_node_map),
                              'num_question_speaker': len(question_speaker_connection),
                              'question_node_mask': question_mask,
                              'question_speaker_mask': question_speaker_mask,
                              'speaker_position_mask': speaker_position_mask})
    pickle.dump(graph_related, open(save_path, 'wb'))


def get_speaker_embedding(example_path, feature_path, save_path):
    # examples = pickle.load(open(example_path, 'rb'), encoding='utf-8')
    # features = pickle.load(open(feature_path, 'rb'), encoding='utf-8')

    examples = torch.load(example_path)
    features = torch.load(feature_path)

    key_speaker_data = []

    for example, feature in tqdm.tqdm(zip(examples, features)):
        speaker_counts = {}
        for utter_dict in example.utterances:
            speaker = ' & '.join(utter_dict['speakers'])
            if speaker not in speaker_counts.keys():
                speaker_counts[speaker] = 1
            else:
                speaker_counts[speaker] += 1
        speaker_id_map = {name: idx + 1 for idx, name in enumerate(speaker_counts.keys())}

        key_utterance = example.key_utterance

        conv_len = len(example.utterances)

        # speaker_range_index = feature.speaker_range_index
        # speaker_position_mask = feature.speaker_position_mask
        input_ids = feature.input_ids
        utterances = example.utterances
        all_utterance_start_pos = []
        tokenizer = ElectraTokenizerFast.from_pretrained('/mnt/sdb/ljn/pretrained/electra-large-discriminator')

        for i in range(len(input_ids)):
            if input_ids[i] == tokenizer.sep_token_id:
                if i == len(input_ids) - 1:
                    continue
                else:
                    all_utterance_start_pos.append(i + 1)
        key_set = set()

        for k in key_utterance:
            max_k = min(k + 3, conv_len)
            for i in range(k, max_k):
                key_set.add(i)
        key_set = sorted(list(key_set))

        key_utter_range = []
        split_point = []
        whole = True
        for i in range(len(key_set)):
            if not whole:
                split_point.append(i)
                whole = True
            first = key_set[i]
            j = min(i + 1, len(key_set) - 1)
            second = key_set[j]
            if second == first + 1:
                whole = True
            else:
                whole = False
        if len(split_point) == 0:
            if len(key_set) > 0:
                key_utter_range.append([key_set[0], key_set[-1]+1])
            else:
                key_utter_range.append([])
        else:
            begin = 0
            for sp in split_point:
                key_utter_range.append([key_set[begin], key_set[sp-1]+1])
                begin = sp
            key_utter_range.append([key_set[begin], key_set[-1]+1])
        speaker_embedding_id = []
        speaker_embedding_mask = []
        picked_utter = []
        for key_utter in key_utter_range:
            speaker_position_id = []
            if len(key_utter) > 0:
                for k in range(key_utter[0], key_utter[1]):
                    speaker_k = ' & '.join(utterances[k]['speakers'])
                    speaker_k_id = speaker_id_map[speaker_k]
                    utterance_k = utterances[k]['utterance']
                    speaker_position_id += [speaker_k_id] * (len(tokenizer.encode(speaker_k + ' : ' + utterance_k, add_special_tokens=False)) + 1)
                considered_context = [all_utterance_start_pos[key_utter[0]], all_utterance_start_pos[key_utter[0]] + len(speaker_position_id)]
                speaker_emb_mask = [0] * all_utterance_start_pos[key_utter[0]] + [1] * len(speaker_position_id)
                speaker_emb_mask += [0] * (512 - len(speaker_emb_mask))
                speaker_embedding_id.append(speaker_position_id)
                speaker_embedding_mask.append(speaker_emb_mask)
                picked_utter.append(considered_context)
        if len(speaker_embedding_id) == 0:
            speaker_embedding_id.append([0])
            speaker_embedding_mask.append([0]*512)
            picked_utter.append([0, 0])
        key_speaker_data.append({'speaker_embedding_id': speaker_embedding_id,
                                 'speaker_embedding_mask': speaker_embedding_mask,
                                 'picked_utter': picked_utter})
    pickle.dump(key_speaker_data, open(save_path, 'wb'))


# get_speaker_embedding('caches2/electra_cache2_512/example_trn_speaker.cache', 'caches2/electra_cache2_512/feature_trn_speaker.cache', 'caches2/electra_cache2_512/key_speaker_trn.pkl')
# get_speaker_embedding('caches2/electra_cache2_512/example_dev_speaker.cache', 'caches2/electra_cache2_512/feature_dev_speaker.cache', 'caches2/electra_cache2_512/key_speaker_dev.pkl')
# get_speaker_embedding('caches2/electra_cache2_512/example_tst_speaker.cache', 'caches2/electra_cache2_512/feature_tst_speaker.cache', 'caches2/electra_cache2_512/key_speaker_tst.pkl')
# def get_coref(key_list, utterances):
#     nlp = spacy.load('en')
#     neurealcoref.add_to_pipe(nlp)
#     blocks = []
#     i = 0
#     while i < len(key_list):
#         if i == len(key_list) - 1:
#             continue
#         else:
#             if key_list[i] + 1 == key_list[i+1]:
#                 blocks.append(key_list[i])
#                 i += 1
#             else:
#                 the_utterance = ''
#                 for k in blocks:
#                     utterance = utterances[k].utterance
#                     the_utterance += utterance
#                 blocks = []
#

get_data_2('caches_graph/electra_cache1_512/example_trn_speaker.pkl',
           'caches_graph/electra_cache1_512/feature_trn_speaker_1.pkl',
           'caches_graph/electra_cache1_512/q_key_utter_rel_trn_graph_ng_3_cr_0.pkl',
           n_gram=3,
           context_range=0,
           question_type_path='speaker_map/question_type.json',
           entities_path='data/friendsqa_trn_ent.pkl',
           speaker_mapping_path='speaker_map/trn_mapping.json')
get_data_2('caches_graph/electra_cache1_512/example_dev_speaker.pkl',
           'caches_graph/electra_cache1_512/feature_dev_speaker_1.pkl',
           'caches_graph/electra_cache1_512/q_key_utter_rel_dev_graph_ng_3_cr_0.pkl',
           n_gram=3,
           context_range=0,
           question_type_path='speaker_map/question_type.json',
           entities_path='data/friendsqa_dev_ent.pkl',
           speaker_mapping_path='speaker_map/dev_mapping.json')
get_data_2('caches_graph/electra_cache1_512/example_tst_speaker.pkl',
           'caches_graph/electra_cache1_512/feature_tst_speaker_1.pkl',
           'caches_graph/electra_cache1_512/q_key_utter_rel_tst_graph_ng_3_cr_0.pkl',
           n_gram=3,
           context_range=0,
           question_type_path='speaker_map/question_type.json',
           entities_path='data/friendsqa_tst_ent.pkl',
           speaker_mapping_path='speaker_map/tst_mapping.json')
