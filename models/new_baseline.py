import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast

from friendsqa.utils.config_1 import *
from friendsqa.utils.utils_split import convert_index_to_text, to_list

_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["start_index", "end_index", "start_log_prob", "end_log_prob"])

# MODEL_CLASSES = {
#     'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
#     'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast)
# }
# TRANSFORMER_CLASS = {'bert': 'bert', 'electra': 'electra'}
# CLS_INDEXES = {'bert': 0, 'electra': 0}
#
# model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MRCModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = 'electra'
        self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
        self.cls_index = 0

        self.electra = ElectraModel(config)

        self.sigmoid = nn.Sigmoid()
        self.predictor = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            output_attentions=False
    ):
        transformer_outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)
        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        logits = self.predictor(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                start_logits = start_logits * p_mask - 65500 * (1 - p_mask)
                end_logits = end_logits * p_mask - 65500 * (1 - p_mask)
            else:
                start_logits = start_logits * p_mask - 1e30 * (1 - p_mask)
                end_logits = end_logits * p_mask - 1e30 * (1 - p_mask)

        if training:
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss
        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)     # shape (bsz, start_n_top)

            end_log_probs = F.softmax(end_logits, dim=1)    # shape (bsz, end_n_top)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=-1)       # shape (bsz, end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            # utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
            # utterance_indices = torch.tensor([i for i in range(128)]).unsqueeze(0).expand(bsz, -1)  # (bsz, utter_num)
            # utteracne_id_expand = utterance_indices.reshape(-1).repeat_interleave(utterance_repeat_num.view(-1).cpu()).view(bsz, -1)  # (bsz, slen)
            # utteracne_id_expand = to_list(utteracne_id_expand)
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        # j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j]
                        end_index = end_top_index[bidx][j]

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
                # span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    # span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs


class MRCModelPosition(nn.Module):
    def __init__(self, config):
        super(MRCModelPosition, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = 'electra'
        self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
        self.cls_index = 0

        self.electra = ElectraModel.from_pretrained(args.model_name)
        self.embeddings = self.electra.embeddings
        self.encoder = self.electra.encoder
        self.conversation_pe = nn.Embedding(config.max_conv_len, config.embedding_size)

        self.sigmoid = nn.Sigmoid()
        self.predictor = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            utterance_position_ids=None,
            conversation_position_ids=None,
            speaker_range_index=None,
            speaker_position_mask=None,
            output_attentions=False
    ):
        if utterance_position_ids is not None:
            # embedding_output = self.embeddings(input_ids=input_ids,
            #                                    position_ids=utterance_position_ids,
            #                                    token_type_ids=token_type_ids)
            input_embedding = self.embeddings.word_embeddings(input_ids)
            token_type_embedding = self.embeddings.token_type_embeddings(token_type_ids)
            position_embedding = self.embeddings.position_embeddings(utterance_position_ids)
        else:
            # embedding_output = self.embeddings(input_ids=input_ids,
            #                                    token_type_ids=token_type_ids)
            position_ids = self.embeddings.position_ids[:, :input_ids.shape[1]]
            input_embedding = self.embeddings.word_embeddings(input_ids)
            token_type_embedding = self.embeddings.token_type_embeddings(token_type_ids)
            position_embedding = self.embeddings.position_embeddings(position_ids)

        if speaker_position_mask is not None:
            speaker_embs = input_embedding.unsqueeze(1) * speaker_position_mask.unsqueeze(3)
            # speaker_embs = embedding_output.unsqueeze(1) * speaker_position_mask.unsqueeze(3)
            # (bsz, spknum, hsz)
            speaker_embs = speaker_embs.sum(2) / (torch.sum(speaker_position_mask, dim=-1, keepdim=True) + 1e-30)
            bsz, slen = speaker_range_index.shape[0], speaker_range_index.shape[1]
            hsz = speaker_embs.shape[2]
            speaker_range_index = speaker_range_index.unsqueeze(2).expand(bsz, slen, hsz)
            speaker_embedding = torch.gather(speaker_embs, -2, speaker_range_index)
            input_embedding += speaker_embedding

        embedding = input_embedding + token_type_embedding + position_embedding

        if conversation_position_ids is not None:
            conv_position_embedding = self.conversation_pe(conversation_position_ids)
            # embedding_output += conv_position_embedding
            embedding += conv_position_embedding
        # embedding_output += speaker_embedding
        embedding = self.embeddings.LayerNorm(embedding)
        embedding_output = self.embeddings.dropout(embedding)

        extended_attention_mask = self.electra.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)

        transformer_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)
        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        logits = self.predictor(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                start_logits = start_logits * p_mask - 65500 * (1 - p_mask)
                end_logits = end_logits * p_mask - 65500 * (1 - p_mask)
            else:
                start_logits = start_logits * p_mask - 1e30 * (1 - p_mask)
                end_logits = end_logits * p_mask - 1e30 * (1 - p_mask)

        if training:
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss
        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)     # shape (bsz, start_n_top)

            end_log_probs = F.softmax(end_logits, dim=1)    # shape (bsz, end_n_top)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=-1)       # shape (bsz, end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        # j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j]
                        end_index = end_top_index[bidx][j]

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
                # span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    # span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs


class MRCModelMask(nn.Module):
    def __init__(self, config):
        super(MRCModelMask, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.5
        self.transformer_name = 'electra'
        self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
        self.cls_index = 0

        self.electra = ElectraModel.from_pretrained(args.model_name)
        self.embeddings = self.electra.embeddings
        self.encoder = self.electra.encoder
        self.conversation_pe = nn.Embedding(config.max_conv_len, config.embedding_size)
        self.add_position_ids = args.add_position_ids

        self.sigmoid = nn.Sigmoid()
        self.predictor = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            utterance_position_ids=None,
            conversation_position_ids=None,
            same_speaker_mask=None,
            different_speaker_mask=None,
            output_attentions=False
    ):
        if self.add_position_ids:
            embedding_output = self.embeddings(input_ids=input_ids,
                                               position_ids=utterance_position_ids,
                                               token_type_ids=token_type_ids)
            # input_embedding = self.embeddings.word_embeddings(input_ids)
            # token_type_embedding = self.embeddings.token_type_embeddings(token_type_ids)
            # position_embedding = self.embeddings.position_embeddings(utterance_position_ids)
        else:
            embedding_output = self.embeddings(input_ids=input_ids,
                                               token_type_ids=token_type_ids)
            # position_ids = self.embeddings.position_ids[:, :input_ids.shape[1]]
            # input_embedding = self.embeddings.word_embeddings(input_ids)
            # token_type_embedding = self.embeddings.token_type_embeddings(token_type_ids)
            # position_embedding = self.embeddings.position_embeddings(position_ids)
        # embedding = input_embedding + token_type_embedding + position_embedding

        if self.add_position_ids:
            conv_position_embedding = self.conversation_pe(conversation_position_ids)
            embedding_output += conv_position_embedding
            # embedding += conv_position_embedding
        # embedding += speaker_embedding
        # embedding = self.embeddings.LayerNorm(embedding)
        # embedding_output = self.embeddings.dropout(embedding)

        # extended_attention_mask = self.electra.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)

        bsz, slen = same_speaker_mask.shape[0], same_speaker_mask.shape[1]
        same_speaker_mask_ = same_speaker_mask.unsqueeze(1).expand(bsz, 8, slen, slen)
        different_speaker_mask_ = different_speaker_mask.unsqueeze(1).expand(bsz, 8, slen, slen)

        extended_attention_mask = torch.cat([same_speaker_mask_, different_speaker_mask_], dim=1)

        # same_speaker_mask_ = same_speaker_mask.unsqueeze(1).expand(bsz, 4, slen, slen)
        # different_speaker_mask_ = different_speaker_mask.unsqueeze(1).expand(bsz, 4, slen, slen)
        # attention_mask_ = attention_mask[:, None, None, :].expand(bsz, 8, slen, slen)
        #
        # extended_attention_mask = torch.stack([same_speaker_mask_, different_speaker_mask_, attention_mask_], dim=1)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        transformer_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)
        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        span_loss_fct = CrossEntropyLoss()

        training = start_pos is not None and end_pos is not None

        logits = self.predictor(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                start_logits = start_logits * p_mask - 65500 * (1 - p_mask)
                end_logits = end_logits * p_mask - 65500 * (1 - p_mask)
            else:
                start_logits = start_logits * p_mask - 1e30 * (1 - p_mask)
                end_logits = end_logits * p_mask - 1e30 * (1 - p_mask)

        if training:
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss
        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1)     # shape (bsz, start_n_top)

            end_log_probs = F.softmax(end_logits, dim=1)    # shape (bsz, end_n_top)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=-1)       # shape (bsz, end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            for bidx in range(bsz):
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        # j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j]
                        end_index = end_top_index[bidx][j]

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
                # span_pred_uid = -1
                best_prob = 0
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                    best_prob = best_one.start_log_prob * best_one.end_log_prob
                    # span_pred_uid = utteracne_id_expand[bidx][best_start_index]
                best_text = best_text.replace(self.tokenizer.sep_token, '')  # in case the long sentence answer
                answer_list.append((qid[bidx], {"answer_text": best_text, "prob": best_prob}))

        outputs = (total_loss, span_loss,) if training else (answer_list,)
        return outputs
