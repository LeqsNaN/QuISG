import os
import json
import torch
import numpy as np
import random
import warnings
from math import sqrt
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import ElectraTokenizerFast
from transformers import ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluate_v2 import main as evaluate_on_squad, EVAL_OPTS
from utils.config import *
# from new_utils import get_dataset, collate_fn
from new_utils_2 import get_dataset, collate_fn
from models.baseline import MRCModel
# , MRCModelSpeaker, MRCModelSpeaker1
from models.position import MRCModelRelPosition, MRCModelRelGraph
from models.position import MRCModelSpeaker
from models.relational_gat import MRCModelRelGATGraph

warnings.filterwarnings("ignore")
device = torch.device("cuda:" + str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, "friendsqa_trn.json")
eval_path = os.path.join(args.data_path, "friendsqa_dev.json")
test_path = os.path.join(args.data_path, "friendsqa_tst.json")


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def train(model_types, model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)
    max_em, max_f1 = 0., 0.
    max_test_em, max_test_f1 = 0., 0.

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    log_file = args.save_path + '/results.txt'
    logss = open(log_file, 'w')

    logss.write(str(args) + '\n\n')

    patience_turns = 0
    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = t_total // (args.epochs * 5)
    steps = 0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask'],
                      'p_mask': batch['p_mask'],
                      'utterance_ids_dict': batch['utterance_ids_dict'],
                      'start_pos': batch['start_pos'],
                      'end_pos': batch['end_pos']}
            # if args.add_speaker_mask:
            #     inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
            if model_types == 'speaker':
                inputs['speaker_position_mask'] = batch['speaker_position_mask']
                # inputs['speaker_range_index'] = batch['speaker_range_index']
                inputs['speaker_embedding_ids'] = batch['speaker_embedding_ids']
                inputs['speaker_embedding_attention_mask'] = batch['speaker_embedding_attention_mask']
                inputs['picked_utterance'] = batch['picked_utter']
            if model_types == 'speaker1':
                inputs['speaker_position_mask'] = batch['speaker_position_mask']
                inputs['speaker_range_index'] = batch['speaker_range_index']
            if model_types == 'position_new':
                inputs['rel_positions'] = batch['rel_positions']
                inputs['rel_attention_mask'] = batch['rel_attention_mask']
                inputs['key_utterance_range'] = batch['key_utterance_range']
                inputs['key_ranges'] = batch['key_ranges']
            if model_types == 'graph':
                inputs['key_utterance_range'] = batch['key_utterance']
                inputs['speaker_position_mask'] = batch['speaker_position_mask']
                inputs['adj'] = batch['adj']
                inputs['word_nodes'] = batch['word_nodes']
                if args.add_utter_mask:
                    inputs['utter_mask'] = batch['utter_mask']
                inputs['related_speaker'] = batch['related_speaker']
                inputs['word_nodes_list'] = batch['num_words']
                inputs['utter_mask_list'] = batch['num_utter']
                inputs['speaker_num_list'] = batch['num_speaker']
                inputs['nodes_mapping'] = batch['nodes_mapping']
                inputs['nodes_mapping_mask'] = batch['nodes_mapping_mask']
                inputs['question_spk_num_list'] = batch['num_question_speaker']
                inputs['question_node_mask'] = batch['question_node_mask']
                inputs['question_speaker_mask'] = batch['question_speaker_mask']
            if model_types == 'rel_graph':
                inputs['key_utterance_range'] = batch['key_utterance']
                inputs['speaker_position_mask'] = batch['speaker_position_mask']
                inputs['adj'] = batch['adj']
                inputs['word_nodes'] = batch['word_nodes']
                if args.add_utter_mask:
                    inputs['utter_mask'] = batch['utter_mask']
                inputs['related_speaker'] = batch['related_speaker']
                inputs['word_nodes_list'] = batch['num_words']
                inputs['utter_mask_list'] = batch['num_utter']
                inputs['speaker_num_list'] = batch['num_speaker']
                inputs['nodes_mapping'] = batch['nodes_mapping']
                inputs['nodes_mapping_mask'] = batch['nodes_mapping_mask']
                inputs['question_spk_num_list'] = batch['num_question_speaker']
                inputs['question_node_mask'] = batch['question_node_mask']
                inputs['question_speaker_mask'] = batch['question_speaker_mask']
                inputs['node_type'] = batch['node_type']
                # inputs['rel_adj'] = batch['rel_adj']
            outputs = model(**inputs)
            loss = outputs[0]
            if args.distributed:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            if t_total is not None:
                scheduler.step()
            if len(outputs) == 4 and args.add_speaker_mask:
                span_loss, utter_loss, speaker_loss = outputs[1].item(), outputs[2].item(), outputs[3].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,UL:%.3f,SpL:%.3f" \
                                     % (loss.item(), span_loss, utter_loss, speaker_loss))
            elif len(outputs) == 3 and not args.add_speaker_mask:
                span_loss, utter_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,UL:%.3f" \
                                     % (loss.item(), span_loss, utter_loss))
            elif len(outputs) == 3 and args.add_speaker_mask:
                span_loss, speaker_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description("Loss:%.3f,SL:%.3f,SpL:%.3f" \
                                     % (loss.item(), span_loss, speaker_loss))
            else:
                span_loss = outputs[1].mean().item()
                pbar.set_description("Loss:%.3f,SL:%.3f" \
                                     % (loss.item(), span_loss))
            model.zero_grad()
            if steps != 0 and steps % logging_step == 0:
                logss.write("Epoch {}, Step {}\n".format(epoch, steps))
                print("Epoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate('dev', model_types, model, eval_dataloader, tokenizer, steps, is_test=False)
                test_result = evaluate('test', model_types, model, test_dataloader, tokenizer, steps, is_test=True)
                print("Eval Result:", eval_result)
                print("Test Result:", test_result)
                eval_em = eval_result['em']
                eval_f1 = eval_result['f1']
                test_em = test_result['em']
                test_f1 = test_result['f1']
                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    max_em = eval_em
                    max_test_em = test_em
                    max_test_f1 = test_f1
                logss.write("    Eval Result: {}\n".format(eval_result))
                logss.write("    Test Result: {}\n".format(test_result))
            steps += 1

    eval_result = evaluate('dev', model_types, model, eval_dataloader, tokenizer, steps, is_test=False)
    test_result = evaluate('test', model_types, model, test_dataloader, tokenizer, steps, is_test=True)
    eval_em = eval_result['em']
    eval_f1 = eval_result['f1']
    test_em = test_result['em']
    test_f1 = test_result['f1']
    if eval_f1 > max_f1:
        max_f1 = eval_f1
        max_em = eval_em
        max_test_em = test_em
        max_test_f1 = test_f1
    print("Eval Result:", eval_result)
    print("Test Result:", test_result)
    logss.write("\nFinal Eval Result: {}\n".format(eval_result))
    logss.write("Final Test Result: {}\n".format(test_result))
    logss.write("\nMax em: {}, Max F1: {}\n".format(round(max_em, 4), round(max_f1, 4)))
    logss.write("Max test em: {}, Max test F1: {}".format(round(max_test_em, 4), round(max_test_f1, 4)))

    logss.close()


def evaluate(eval_type, model_types, model, eval_loader, tokenizer, steps, is_test=False):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    answer_dict, na_dict = {}, {}
    correct_num, all_num = 0, 0
    target_file_path = 'data/' + ('tst' if is_test else 'dev') + '_uids_target.json'
    with open(target_file_path, "r") as f:
        target_uids_dict = json.load(f)

    for _, batch in pbar:
        cur_batch_size = len(batch['input_ids'])

        inputs = {'input_ids': batch['input_ids'],
                  'token_type_ids': batch['token_type_ids'],
                  'attention_mask': batch['attention_mask'],
                  'p_mask': batch['p_mask'],
                  'context': batch['context'],
                  'utterance_ids_dict': batch['utterance_ids_dict'],
                  'offset_mapping': batch['offset_mapping'],
                  'qid': batch['qid']
                  }
        # if args.add_speaker_mask:
        #     inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
        if model_types == 'speaker':
            inputs['speaker_position_mask'] = batch['speaker_position_mask']
            # inputs['speaker_range_index'] = batch['speaker_range_index']
            inputs['speaker_embedding_ids'] = batch['speaker_embedding_ids']
            inputs['speaker_embedding_attention_mask'] = batch['speaker_embedding_attention_mask']
            inputs['picked_utterance'] = batch['picked_utter']
        if model_types == 'speaker1':
            inputs['speaker_position_mask'] = batch['speaker_position_mask']
            inputs['speaker_range_index'] = batch['speaker_range_index']
        if model_types == 'position_new':
            inputs['rel_positions'] = batch['rel_positions']
            inputs['rel_attention_mask'] = batch['rel_attention_mask']
            inputs['key_utterance_range'] = batch['key_utterance_range']
            inputs['key_ranges'] = batch['key_ranges']
        if model_types == 'graph':
            inputs['key_utterance_range'] = batch['key_utterance']
            inputs['speaker_position_mask'] = batch['speaker_position_mask']
            inputs['adj'] = batch['adj']
            inputs['word_nodes'] = batch['word_nodes']
            if args.add_utter_mask:
                inputs['utter_mask'] = batch['utter_mask']
            inputs['related_speaker'] = batch['related_speaker']
            inputs['word_nodes_list'] = batch['num_words']
            inputs['utter_mask_list'] = batch['num_utter']
            inputs['speaker_num_list'] = batch['num_speaker']
            inputs['nodes_mapping'] = batch['nodes_mapping']
            inputs['nodes_mapping_mask'] = batch['nodes_mapping_mask']
            inputs['question_spk_num_list'] = batch['num_question_speaker']
            inputs['question_node_mask'] = batch['question_node_mask']
            inputs['question_speaker_mask'] = batch['question_speaker_mask']
        if model_types == 'rel_graph':
            inputs['key_utterance_range'] = batch['key_utterance']
            inputs['speaker_position_mask'] = batch['speaker_position_mask']
            inputs['adj'] = batch['adj']
            inputs['word_nodes'] = batch['word_nodes']
            if args.add_utter_mask:
                inputs['utter_mask'] = batch['utter_mask']
            inputs['related_speaker'] = batch['related_speaker']
            inputs['word_nodes_list'] = batch['num_words']
            inputs['utter_mask_list'] = batch['num_utter']
            inputs['speaker_num_list'] = batch['num_speaker']
            inputs['nodes_mapping'] = batch['nodes_mapping']
            inputs['nodes_mapping_mask'] = batch['nodes_mapping_mask']
            inputs['question_spk_num_list'] = batch['num_question_speaker']
            inputs['question_node_mask'] = batch['question_node_mask']
            inputs['question_speaker_mask'] = batch['question_speaker_mask']
            inputs['node_type'] = batch['node_type']
            # inputs['rel_adj'] = batch['rel_adj']
        outputs = model(**inputs)
        answer_list = outputs[0]
        # if args.add_speaker_mask:
        #     b_correct_num, b_all_num = outputs[1]
        #     correct_num += b_correct_num
        #     all_num += b_all_num
        for qid, ans_record in answer_list:
            real_qid = qid.split('-')[0]
            offset = int(qid.split('-')[1])
            ans_record['span_pred_uid'] += offset
            if 'model_pred_uid' in ans_record.keys(): ans_record['model_pred_uid'] += offset
            if real_qid not in answer_dict.keys():
                answer_dict[real_qid] = ans_record
            else:
                cur_best_prob = answer_dict[real_qid]['prob']
                if ans_record['prob'] > cur_best_prob:
                    answer_dict[real_qid] = ans_record
    # computing utterance matching (UM)
    assert len(answer_dict) == len(target_uids_dict)
    all_example_num, model_pred_correct_num, span_um_num = len(answer_dict), 0, 0
    for qid, target_uids in target_uids_dict.items():
        ans_record = answer_dict[qid]
        span_um_num += 1 if ans_record['span_pred_uid'] in target_uids else 0
        if 'model_pred_uid' in ans_record.keys():
            model_pred_correct_num += 1 if ans_record['model_pred_uid'] in target_uids else 0
    model_um = model_pred_correct_num / all_example_num
    span_um = span_um_num / all_example_num

    # computing f1 and em using official SQuAD transcript
    answer_dict = {qid: ans_record['answer_text'] for qid, ans_record in answer_dict.items()}
    preds_file = args.pred_file.split('.')[:1] + ['_' + eval_type + '_' + str(steps) + '.json']
    preds_file = ''.join(preds_file)
    with open(preds_file, "w") as f:
        json.dump(answer_dict, f, indent=2)
    # if args.add_speaker_mask:
    #     print("Speaker prediction acc: %.5f" % (correct_num / all_num))
    evaluate_options = EVAL_OPTS(data_file=test_path if is_test else eval_path,
                                 pred_file=preds_file,
                                 na_prob_file=None)
    res = evaluate_on_squad(evaluate_options)
    em = res['exact']
    f1 = res['f1']
    rtv_dict = {'em': em, 'f1': f1, 'um': span_um, 'model_um': model_um}
    model.train()

    return rtv_dict


if __name__ == "__main__":
    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group('nccl', init_method='env://')
    #     device = torch.device(f'cuda:{args.local_rank}')
    set_seed()
    model_kind = args.model_kind
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
    config = ElectraConfig.from_pretrained(args.model_name)
    if args.model_type != 'xlnet':
        config.start_n_top = 5
        config.end_n_top = 5

    # training
    # train_dataset = get_dataset(train_path, args.cache_path, tokenizer, args.max_length,
    #                             training=True, key_utterance_path='data/friendsqa_qc_bert_train.json',
    #                             graph_path='caches1/electra_cache1_512/trn_graph_cr_' + str(args.context_range) + '_ng_' + str(args.n_gram) + '.pkl')
    # eval_dataset = get_dataset(eval_path, args.cache_path, tokenizer, args.max_length,
    #                            training=False, key_utterance_path='data/friendsqa_qc_bert_dev.json',
    #                            graph_path='caches1/electra_cache1_512/dev_graph_cr_' + str(args.context_range) + '_ng_' + str(args.n_gram) + '.pkl')
    # test_dataset = get_dataset(test_path, args.cache_path, tokenizer, args.max_length,
    #                            training=False, key_utterance_path='data/friendsqa_qc_bert_test.json',
    #                            graph_path='caches1/electra_cache1_512/tst_graph_cr_' + str(args.context_range) + '_ng_' + str(args.n_gram) + '.pkl')

    if args.model_kind == 'graph' or args.model_kind == 'rel_graph':
        train_dataset = get_dataset(train_path, args.cache_path, tokenizer, args.max_length,
                                    training=True, key_utterance_path='Train_key.json',
                                    graph_path='caches_graph/electra_cache1_512/q_key_utter_rel_trn_graph_ng_' + str(args.n_gram) + '_cr_' + str(args.context_range) + '.pkl')
        eval_dataset = get_dataset(eval_path, args.cache_path, tokenizer, args.max_length,
                                   training=False, key_utterance_path='Dev_key.json',
                                   graph_path='caches_graph/electra_cache1_512/q_key_utter_rel_dev_graph_ng_' + str(args.n_gram) + '_cr_' + str(args.context_range) + '.pkl')
        test_dataset = get_dataset(test_path, args.cache_path, tokenizer, args.max_length,
                                   training=False, key_utterance_path='Test_key.json',
                                   graph_path='caches_graph/electra_cache1_512/q_key_utter_rel_tst_graph_ng_' + str(args.n_gram) + '_cr_' + str(args.context_range) + '.pkl')
    else:
        train_dataset = get_dataset(train_path, args.cache_path, tokenizer, args.max_length,
                                    training=True, key_utterance_path=None,
                                    graph_path=None, key_speaker_path='caches2/electra_cache2_512/key_speaker_trn.pkl')
        eval_dataset = get_dataset(eval_path, args.cache_path, tokenizer, args.max_length,
                                   training=False, key_utterance_path=None,
                                   graph_path=None, key_speaker_path='caches2/electra_cache2_512/key_speaker_dev.pkl')
        test_dataset = get_dataset(test_path, args.cache_path, tokenizer, args.max_length,
                                   training=False, key_utterance_path=None,
                                   graph_path=None, key_speaker_path='caches2/electra_cache2_512/key_speaker_tst.pkl')

    # if args.distributed:
    #     train_sampler = DistributedSampler(train_dataset)
    #     eval_sampler = DistributedSampler(eval_dataset)
    #     test_sampler = DistributedSampler(test_dataset)
    # else:
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    if model_kind == 'baseline':
        model = MRCModel.from_pretrained(args.model_name, config=config)
    # elif model_kind == 'speaker1':
    #     model = MRCModelSpeaker1.from_pretrained(args.model_name, config=config)
    elif model_kind == 'position_new':
        model = MRCModelRelPosition.from_pretrained(args.model_name, config=config)
    elif model_kind == 'graph':
        model = MRCModelRelGraph.from_pretrained(args.model_name, config=config)
    elif model_kind == 'rel_graph':
        model = MRCModelRelGATGraph.from_pretrained(args.model_name, config=config)
    else:
        model = MRCModelSpeaker.from_pretrained(args.model_name, config=config)
    if args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    model = model.cuda()

    train(model_kind, model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)
