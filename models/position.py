import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import BertLayer
from torch.nn.utils.rnn import pad_sequence
from friendsqa.models.baseline import PoolerStartLogits, PoolerEndLogits

from friendsqa.utils.config import *
from friendsqa.utils.utils_split import convert_index_to_text, to_list

_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["start_index", "end_index", "start_log_prob", "end_log_prob"])

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'electra': 'electra'}
CLS_INDEXES = {'bert': 0, 'electra': 0}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x1, x2, mask=None):
        z = torch.sigmoid(self.w1(x1) + self.w2(x2))
        output = x1 * (-z + 1) + x2 * z
        if mask is not None:
            output = output * mask + x2 * (- mask + 1)
        return output


class MRCModelRelPosition(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.cls_index = CLS_INDEXES[args.model_type]

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.rel_position_embeddings = nn.Embedding(10, 1024, padding_idx=0)
        self.gate = Gate(1024)

        self.key_utter_encoder = nn.ModuleList()
        self.ku_layers = args.ku_layers
        self.trm_layers = args.trm_layers
        for i in range(self.ku_layers):
            self.key_utter_encoder.append(BertLayer(config))
        self.further_encoder = nn.ModuleList()
        for i in range(self.trm_layers):
            self.further_encoder.append(BertLayer(config))

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            utterance_ids_dict=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            output_attentions=False,
            rel_positions=None,
            rel_attention_mask=None,
            key_utterance_range=None,
            key_ranges=None
    ):
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)

        bsz, slen, hsz = hidden_states.shape

        text_length = rel_positions.shape[2]
        key_utterances_batch_ = []
        for hidden_state, key_range in zip(hidden_states, key_utterance_range):
            batch_utterances = []
            for key in key_range:
                start_token_pos = key[0].item()
                end_token_pos = key[1].item()
                picked_utters = hidden_state[start_token_pos: end_token_pos]
                batch_utterances.append(picked_utters)
            batch_utterances = pad_sequence(batch_utterances, batch_first=True)
            key_utterances_batch_.append(batch_utterances)
        # [bsz*4, slen, hsz]
        key_utterances_batch = [torch.cat([k, torch.zeros(k.shape[0], text_length-k.shape[1], hsz, device=k.device)], dim=1) for k in key_utterances_batch_]
        key_utterances_batch = torch.cat(key_utterances_batch, dim=0)
        rel_positions = rel_positions.view(bsz*4, -1)
        rel_attention_mask = rel_attention_mask.view(bsz*4, -1)
        rel_pos_embeddings = self.rel_position_embeddings(rel_positions)
        key_utterances_batch = key_utterances_batch + rel_pos_embeddings

        extended_rel_attention_mask = transformer.get_extended_attention_mask(rel_attention_mask,
                                                                              input_shape=rel_positions.size(),
                                                                              device=hidden_states.device)

        for i in range(self.ku_layers):
            key_utterances_batch = self.key_utter_encoder[i](hidden_states=key_utterances_batch,
                                                             attention_mask=extended_rel_attention_mask)[0]
        key_utterances_batch = key_utterances_batch.view(bsz, 4, -1, hsz)

        batch_count = 0
        for key_utterances, key_range in zip(key_utterances_batch, key_utterance_range):
            for j in range(key_range.shape[0]-1, -1, -1):
                key = key_range[j]
                if key[0].item() == key[1].item() == 0:
                    continue
                else:
                    updated_key_utterances = self.gate(key_utterances[j, :key[1].item()-key[0].item(), :], hidden_states[batch_count, key[0].item(): key[1].item(), :].clone())
                    hidden_states[batch_count, key[0].item(): key[1].item(), :] = updated_key_utterances
            batch_count += 1

        extended_attention_mask = transformer.get_extended_attention_mask(attention_mask,
                                                                          input_shape=input_ids.size(),
                                                                          device=hidden_states.device)

        for i in range(self.trm_layers):
            hidden_states = self.further_encoder[i](hidden_states=hidden_states, attention_mask=extended_attention_mask)[0]

        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        start_logits = self.start_predictor(hidden_states, p_mask=key_ranges)  # (bsz, seqlen)

        if training:
            end_logits = self.end_predictor(hidden_states, start_positions=start_pos, p_mask=key_ranges)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            # if key_ranges is not None:
            #     start_log_probs = start_log_probs * key_ranges
            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(hidden_states_expanded, start_states=start_states, p_mask=key_ranges.unsqueeze(-1))
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)
            # if key_ranges is not None:
            #     end_log_probs = end_log_probs * key_ranges.unsqueeze(-1)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
            utterance_indices = torch.tensor([i for i in range(128)]).unsqueeze(0).expand(bsz, -1)  # (bsz, utter_num)
            utteracne_id_expand = utterance_indices.reshape(-1).repeat_interleave(utterance_repeat_num.view(-1).cpu()).view(bsz, -1)  # (bsz, slen)
            utteracne_id_expand = to_list(utteracne_id_expand)
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob, "span_pred_uid": span_pred_uid}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs


class MRCModelRelGraph(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.nfeat = args.nfeat
        self.nhid = args.nhid
        self.nheads = args.nheads
        self.alpha = args.alpha
        self.gat_dropout = args.gat_dropout
        self.gat_num_layer = args.gat_num_layer
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.cls_index = CLS_INDEXES[args.model_type]

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        # self.rel_position_embeddings = nn.Embedding(10, 1024, padding_idx=0)
        self.gate = Gate(1024)

        self.gat_layers = nn.ModuleList()
        for i in range(self.gat_num_layer):
            self.gat_layers.append(GAT(self.nfeat, self.nhid, self.nheads,
                                       self.gat_dropout, self.alpha))
            # self.gat_layers.append(GATFFN(self.nfeat, self.nhid, self.nheads,
            #                               self.gat_dropout, self.alpha))
            # self.gat_layers.append(BertLayer(config))

        self.trm_layers = args.trm_layers
        self.further_encoder = nn.ModuleList()
        for i in range(self.trm_layers):
            self.further_encoder.append(BertLayer(config))

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            utterance_ids_dict=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            output_attentions=False,
            key_utterance_range=None,
            speaker_position_mask=None,
            adj=None,
            word_nodes=None,
            utter_mask=None,
            related_speaker=None,
            word_nodes_list=None,
            utter_mask_list=None,
            speaker_num_list=None,
            nodes_mapping=None,
            nodes_mapping_mask=None,
            question_node_mask=None,
            question_speaker_mask=None,
            question_spk_num_list=None
    ):
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)

        bsz, slen, hsz = hidden_states.shape

        speaker_embs = hidden_states.unsqueeze(1) * speaker_position_mask.unsqueeze(3)
        # (bsz, spknum, hsz)
        speaker_embs = speaker_embs.sum(2) / (torch.sum(speaker_position_mask, dim=-1, keepdim=True) + 1e-30)

        if utter_mask is None:
            utter_embs = None
        else:
            utter_embs = hidden_states.unsqueeze(1) * utter_mask.unsqueeze(3)
            # (bsz, uttnum, hsz)
            utter_embs = utter_embs.sum(2) / (torch.sum(utter_mask, dim=-1, keepdim=True) + 1e-30)

        picked_slen = word_nodes.shape[1]
        word_nodes_ = word_nodes.unsqueeze(2).expand(bsz, picked_slen, hsz)
        # (bsz, picked_slen, hsz)
        word_embs = torch.gather(hidden_states.clone(), 1, word_nodes_)

        question_part_hidden_states = hidden_states[:, :32, :]

        question_embs = question_part_hidden_states * question_node_mask.unsqueeze(2)
        # (bsz, hsz)
        question_embs = question_embs.sum(1) / (torch.sum(question_node_mask, dim=-1, keepdim=True) + 1e-30)

        # (bsz, snum, qlen)
        question_speaker_embs = question_part_hidden_states.unsqueeze(1) * question_speaker_mask.unsqueeze(3)
        # (bsz, snum, hsz)
        question_speaker_embs = question_speaker_embs.sum(2) / (torch.sum(question_speaker_mask, dim=-1, keepdim=True) + 1e-30)

        max_length = adj.shape[1]

        nodes_emb = []
        i = 0
        for wl, ul, rs, sl, ql in zip(word_nodes_list, utter_mask_list, related_speaker, speaker_num_list, question_spk_num_list):
            wnode = word_embs[i, :wl[0], :]
            # unode = utter_embs[i, :ul[0], :].clone()
            if utter_embs is None:
                unode = None
            else:
                unode = utter_embs[i, :ul[0], :]
            num_s = rs.shape[0]
            snode = torch.gather(speaker_embs[i].clone(), 0, rs.unsqueeze(1).expand(num_s, hsz))[:sl[0], :]
            qnode = question_embs[i:i+1, :]
            qsnode = question_speaker_embs[i, :ql[0], :]
            if unode is None:
                node = torch.cat([wnode, snode, qnode, qsnode], dim=0)
            else:
                node = torch.cat([wnode, snode, unode, qnode, qsnode], dim=0)
            nodes_emb.append(node)
            i += 1
        nodes_emb = pad_sequence(nodes_emb, batch_first=True)
        if nodes_emb.shape[1] < max_length:
            nodes_emb = torch.cat([nodes_emb, torch.zeros(nodes_emb.shape[0], max_length-nodes_emb.shape[1], nodes_emb.shape[2]).to(nodes_emb.device)], dim=1)

        # extended_adj = (1 - adj[:, None, :, :]) * -10000.0

        for i in range(self.gat_num_layer):
            # [bsz, num_nodes, hidden_size]
            nodes_emb = self.gat_layers[i](nodes_emb, adj)
            # nodes_emb = self.gat_layers[i](nodes_emb, attention_mask=extended_adj)[0]
        batch_count = 0

        for nodes, node_mapping, node_mask, key_range in zip(nodes_emb, nodes_mapping, nodes_mapping_mask, key_utterance_range):
            if not key_range.item():
                batch_count += 1
                continue
            else:
                first_dim = node_mapping.shape[0]
                nodes_scattered = torch.gather(nodes, 0, node_mapping.unsqueeze(1).expand(first_dim, nodes.shape[1]))
                nodes_scattered = nodes_scattered * node_mask.unsqueeze(1)
                hidden_states[batch_count] = nodes_scattered + hidden_states[batch_count].clone()
                # hidden_states[batch_count] = self.gate(nodes_scattered, hidden_states[batch_count].clone(), node_mask.unsqueeze(1))
                batch_count += 1

        extended_attention_mask = transformer.get_extended_attention_mask(attention_mask,
                                                                          input_shape=input_ids.size(),
                                                                          device=hidden_states.device)

        for i in range(self.trm_layers):
            hidden_states = self.further_encoder[i](hidden_states=hidden_states, attention_mask=extended_attention_mask)[0]

        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        start_logits = self.start_predictor(hidden_states, p_mask=p_mask)  # (bsz, seqlen)

        if training:
            if args.hard:
                start_logits = start_logits * nodes_mapping_mask.add(args.hard_rate).clamp(max=1)
            end_logits = self.end_predictor(hidden_states, start_positions=start_pos, p_mask=p_mask)
            if args.hard:
                end_logits = end_logits * nodes_mapping_mask.add(args.hard_rate).clamp(max=1)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss
        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            if args.hard_before_softmax:
                key_utter_soft_mask = nodes_mapping_mask.add(args.eval_hard_rate).clamp(max=1)
                start_logits = start_logits * key_utter_soft_mask

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            if args.hard:
                key_utter_soft_mask = nodes_mapping_mask.add(args.eval_hard_rate).clamp(max=1)
                start_log_probs = start_log_probs * key_utter_soft_mask
                # start_log_probs = start_log_probs * nodes_mapping_mask
            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            if args.hard_before_softmax:
                key_utter_soft_mask = nodes_mapping_mask.add(args.eval_hard_rate).clamp(max=1)
                end_logits = end_logits * key_utter_soft_mask.unsqueeze(2)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            if args.hard:
                key_utter_soft_mask = nodes_mapping_mask.add(args.eval_hard_rate).clamp(max=1)
                end_log_probs = end_log_probs * key_utter_soft_mask.unsqueeze(2)
                # end_log_probs = end_log_probs * nodes_mapping_mask

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            # start_top_index = to_list(start_top_index)
            start_top_index = start_top_index.data.cpu().numpy().tolist()
            # start_top_log_probs = to_list(start_top_log_probs)
            start_top_log_probs = start_top_log_probs.data.cpu().numpy().tolist()
            # end_top_index = to_list(end_top_index)
            end_top_index = end_top_index.data.cpu().numpy().tolist()
            # end_top_log_probs = to_list(end_top_log_probs)
            end_top_log_probs = end_top_log_probs.data.cpu().numpy().tolist()

            answer_list = []
            utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
            utterance_indices = torch.tensor([i for i in range(128)]).unsqueeze(0).expand(bsz, -1)  # (bsz, utter_num)
            utteracne_id_expand = utterance_indices.reshape(-1).repeat_interleave(utterance_repeat_num.view(-1).cpu()).view(bsz, -1)  # (bsz, slen)
            # utteracne_id_expand = to_list(utteracne_id_expand)
            utteracne_id_expand = utteracne_id_expand.data.cpu().numpy().tolist()
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob, "span_pred_uid": span_pred_uid}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs


class GATFFN(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha):
        super(GATFFN, self).__init__()
        self.gat = GAT(nfeat, nhid, nheads, dropout, alpha)
        self.linear1 = nn.Linear(nfeat, 4096)
        self.linear2 = nn.Linear(4096, nfeat)
        self.norm = nn.LayerNorm(nfeat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.gat(x, adj)
        linear_out = self.linear2(F.relu(self.linear1(x)))
        output = self.norm(self.dropout(linear_out) + x)
        return output


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(1, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(1, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)   # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)    # e.shape: (B, N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B*nheads, N, out_feature)
        # self.a.shape (1, 2 * out_feature, 1)
        # Wh1&2.shape (B*nheads, N, 1)
        # e.shape (B*nheads, N, N)
        Wh1 = torch.matmul(Wh, self.a[:, :self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[:, self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MRCModelSpeaker(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.cls_index = CLS_INDEXES[args.model_type]

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.gate = Gate(1024)

        self.key_utter_encoder = nn.ModuleList()
        self.ku_layers = args.ku_layers
        self.trm_layers = args.trm_layers
        for i in range(self.ku_layers):
            self.key_utter_encoder.append(BertLayer(config))
        self.further_encoder = nn.ModuleList()
        for i in range(self.trm_layers):
            self.further_encoder.append(BertLayer(config))

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            utterance_ids_dict=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            output_attentions=False,
            speaker_position_mask=None,
            speaker_embedding_ids=None,
            speaker_embedding_attention_mask=None,
            picked_utterance=None
    ):
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)

        bsz, slen, hsz = hidden_states.shape

        speaker_embs = hidden_states.unsqueeze(1) * speaker_position_mask.unsqueeze(3)
        # (bsz, spknum, hsz)
        speaker_embs = speaker_embs.sum(2) / (torch.sum(speaker_position_mask, dim=-1, keepdim=True) + 1e-30)

        picked_num_tokens = speaker_embedding_ids.shape[2]

        picked_tokens = []
        for hidden_state, picked_utter in zip(hidden_states, picked_utterance):
            picked_token = []
            for pu in picked_utter:
                start_token_pos = pu[0].item()
                end_token_pos = pu[1].item()
                picked_token.append(torch.cat([hidden_state[start_token_pos: end_token_pos],
                                               torch.zeros(picked_num_tokens+start_token_pos-end_token_pos, hsz).to(hidden_state.device)], dim=0))
            # picked_token = pad_sequence(picked_token, batch_first=True)
            picked_token = torch.stack(picked_token, dim=0)
            picked_tokens.append(picked_token)
        # [bsz*num_block, num_token, hsz]
        picked_tokens = torch.stack(picked_tokens, dim=0).view(-1, picked_num_tokens, hsz)
        sid_shape0, sid_shape1, sid_shape2 = speaker_embedding_ids.shape
        speaker_embedding_ids = speaker_embedding_ids.unsqueeze(3).expand(sid_shape0, sid_shape1, sid_shape2, hsz)
        speaker_embs_ = speaker_embs.unsqueeze(1).expand(speaker_embs.shape[0], sid_shape1, speaker_embs.shape[1], speaker_embs.shape[2])
        speaker_embs_ = speaker_embs_.contiguous().view(-1, speaker_embs.shape[1], speaker_embs.shape[2])
        picked_speaker_embs = torch.gather(speaker_embs_, 1, speaker_embedding_ids.view(-1, sid_shape2, hsz))

        picked_tokens = picked_tokens + picked_speaker_embs

        expanded_spk_attention_mask = transformer.get_extended_attention_mask(speaker_embedding_attention_mask.view(-1, sid_shape2),
                                                                              input_shape=speaker_embedding_attention_mask.size(),
                                                                              device=picked_tokens.device)
        for i in range(self.ku_layers):
            picked_tokens = self.key_utter_encoder[i](hidden_states=picked_tokens,
                                                      attention_mask=expanded_spk_attention_mask)[0]
        picked_tokens = picked_tokens.view(sid_shape0, sid_shape1, sid_shape2, -1)

        batch_count = 0
        for key_utterances, key_range in zip(picked_tokens, picked_utterance):
            for j in range(key_range.shape[0]):
                key = key_range[j]
                if key[0].item() == key[1].item() == 0:
                    continue
                else:
                    # updated_key_utterances = self.gate(key_utterances[j, :key[1].item() - key[0].item(), :], hidden_states[batch_count, key[0].item(): key[1].item(), :].clone())
                    updated_key_utterances = key_utterances[j, :key[1].item() - key[0].item(), :] + hidden_states[batch_count, key[0].item(): key[1].item(), :].clone()
                    hidden_states[batch_count, key[0].item(): key[1].item(), :] = updated_key_utterances
            batch_count += 1

        extended_attention_mask = transformer.get_extended_attention_mask(attention_mask,
                                                                          input_shape=input_ids.size(),
                                                                          device=hidden_states.device)

        for i in range(self.trm_layers):
            hidden_states = self.further_encoder[i](hidden_states=hidden_states, attention_mask=extended_attention_mask)[0]

        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        start_logits = self.start_predictor(hidden_states, p_mask=p_mask)  # (bsz, seqlen)

        if training:
            end_logits = self.end_predictor(hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            # if key_ranges is not None:
            #     start_log_probs = start_log_probs * key_ranges
            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)
            # if key_ranges is not None:
            #     end_log_probs = end_log_probs * key_ranges.unsqueeze(-1)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
            utterance_indices = torch.tensor([i for i in range(128)]).unsqueeze(0).expand(bsz, -1)  # (bsz, utter_num)
            utteracne_id_expand = utterance_indices.reshape(-1).repeat_interleave(utterance_repeat_num.view(-1).cpu()).view(bsz, -1)  # (bsz, slen)
            utteracne_id_expand = to_list(utteracne_id_expand)
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob, "span_pred_uid": span_pred_uid}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs
