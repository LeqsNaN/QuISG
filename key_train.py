import os
import torch
import numpy as np
import random
from key_model import KeyModel
from key_utils import get_dataloader
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import ElectraTokenizerFast, ElectraConfig
from tqdm import tqdm
from utils.config import *


train_path = os.path.join(args.data_path, "friendsqa_trn.json")
eval_path = os.path.join(args.data_path, "friendsqa_dev.json")
test_path = os.path.join(args.data_path, "friendsqa_tst.json")


def train(model, train_loader, dev_loader, test_loader):
    max_recall = 0
    max_test_recall = 0
    max_precision = 0
    max_test_precision = 0
    max_f1 = 0
    max_test_f1 = 0

    log_file = 'similarities/results' + str(args.index) + '.txt'
    logss = open(log_file, 'w')

    logss.write(str(args) + '\n\n')

    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    # scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = t_total // (args.epochs * 5)
    steps = 0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'utterance_range': batch['utterance_mask'],
                      'question_mask': batch['question_mask'],
                      'label': batch['label'],
                      'label_mask': batch['label_mask']}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            all_optimizer.step()
            # if t_total is not None:
            #     scheduler.step()
            pbar.set_description("Loss:%.3f" % (loss.item()))
            model.zero_grad()
            if steps != 0 and steps % logging_step == 0:
                logss.write("Epoch {}, Step {}\n".format(epoch, steps))
                print("Epoch {}, Step {}".format(epoch, steps))
                eval_recall, eval_precision, eval_f1 = evaluate('dev', model, dev_loader)
                test_recall, test_precision, test_f1 = evaluate('test', model, test_loader)
                print("Eval Result: Recall {}, Precision {}, F1 {}".format(eval_recall, eval_precision, eval_f1))
                print("Test Result: Recall {}, Precision {}, F1 {}".format(test_recall, test_precision, test_f1))
                if eval_recall > max_recall:
                    max_recall = eval_recall
                    max_precision = eval_precision
                    max_f1 = eval_f1
                    max_test_recall = test_recall
                    max_test_precision = test_precision
                    max_test_f1 = test_f1
                    torch.save(model, 'similarities/model' + str(args.index) + '.pkl')
                logss.write("    Eval Result: Recall {}, Precision {}, F1 {}\n".format(eval_recall, eval_precision, eval_f1))
                logss.write("    Test Result: Recall {}, Precision {}, F1 {}\n".format(test_recall, test_precision, test_f1))
            steps += 1
    eval_recall, eval_precision, eval_f1 = evaluate('dev', model, dev_loader)
    test_recall, test_precision, test_f1 = evaluate('test', model, test_loader)
    if eval_recall > max_recall:
        max_recall = eval_recall
        max_precision = eval_precision
        max_f1 = eval_f1
        max_test_recall = test_recall
        max_test_precision = test_precision
        max_test_f1 = test_f1
        torch.save(model, 'similarities/model' + str(args.index) + '.pkl')
    print("Eval Result: Recall {}, Precision {}, F1 {}".format(eval_recall, eval_precision, eval_f1))
    print("Test Result: Recall {}, Precision {}, F1 {}".format(test_recall, test_precision, test_f1))
    logss.write("Last Eval Result: Recall {}, Precision {}, F1 {}\n".format(eval_recall, eval_precision, eval_f1))
    logss.write("Last Test Result: Recall {}, Precision {}, F1 {}\n".format(test_recall, test_precision, test_f1))

    logss.write("\nFinal Eval Result: Recall {}, Precision {}, F1 {}\n".format(max_recall, max_precision, max_f1))
    logss.write("Final Test Result: Recall {}, Precision {}, F1 {}\n".format(max_test_recall, max_test_precision, max_test_f1))

    logss.close()


def evaluate(eval_type, model, eval_dataloader):
    model.eval()

    preds = []
    gold = []
    masks =[]
    pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))

    with torch.no_grad():
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'utterance_range': batch['utterance_mask'],
                      'question_mask': batch['question_mask'],
                      'label': batch['label'],
                      'label_mask': batch['label_mask']}
            outputs = model(**inputs)
            logits = outputs[1]
            pred = logits.gt(0.5).view(-1).long().cpu().numpy()
            preds.append(pred)
            gold.append(batch['label'].data.view(-1).cpu().long().numpy())
            masks.append(batch['label_mask'].data.view(-1).cpu().long().numpy())
    preds = np.concatenate(preds)
    gold = np.concatenate(gold)
    masks = np.concatenate(masks)
    # accuracy = round(accuracy_score(gold, preds), 4)
    recall = target_recall(gold, preds, masks)
    precision = target_precision(gold, preds, masks)
    f1 = (2 * precision * recall) / (precision + recall)

    model.train()
    return round(recall, 4), round(precision, 4), round(f1, 4)


def find_key_utterance(eval_type, model, eval_dataloader, topk):
    model.eval()

    # preds = []
    # gold = []
    # masks = []
    pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))

    all_question = 0
    right_question = 0
    right_question_topk = 0

    with torch.no_grad():
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'utterance_range': batch['utterance_mask'],
                      'question_mask': batch['question_mask'],
                      'label': batch['label'],
                      'label_mask': batch['label_mask']}
            key_utterance = batch['key_utterance'][0]
            outputs = model(**inputs)
            logits = outputs[1]
            conv_len = logits.shape[0]
            utterance_range = []
            if conv_len <= 3:
                utterance_range.append([0, conv_len])
            else:
                for i in range(conv_len-2):
                    utterance_range.append([i, i+3])
            pred = logits.gt(0.5).view(-1).long().cpu().numpy()
            find_answer = False
            all_question += 1
            for i, p in enumerate(pred):
                if p == 1:
                    utter_range = utterance_range[i]
                    for answer in key_utterance:
                        if answer in list(range(utter_range[0], utter_range[1])):
                            right_question += 1
                            find_answer = True
                            break
                    if find_answer:
                        break
            topk_logits, topk_index = torch.topk(logits, k=topk, dim=-1)
            topk_pred = topk_logits.gt(0.5).view(-1).long().cpu().numpy()
            find_answer = False
            for i, p in enumerate(topk_pred):
                if p == 1:
                    utter_range = utterance_range[topk_index[i]]
                    for answer in key_utterance:
                        if answer in list(range(utter_range[0], utter_range[1])):
                            right_question_topk += 1
                            find_answer = True
                            break
                    if find_answer:
                        break
    recall_all = round(right_question/all_question, 4)
    recall_topk = round(right_question_topk/all_question, 4)
    print('{}: Recall All: {}, Recall Topk: {}'.format(eval_type, recall_all, recall_topk))


def target_recall(y_true, y_pred, mask=None):
    golden_num = 0
    right_num = 0
    for i, j, k in zip(y_true, y_pred, mask):
        if k == 1:
            if i == 1:
                golden_num += 1
                if j == i:
                    right_num += 1
    return right_num / golden_num


def target_precision(y_true, y_pred, mask=None):
    pred_num = 0
    right_num = 0
    for i, j, k in zip(y_true, y_pred, mask):
        if k == 1:
            if j == 1:
                pred_num += 1
                if i == j:
                    right_num += 1
    return right_num / pred_num


def main():
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
    config = ElectraConfig.from_pretrained(args.model_name)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.eval_key_utterance:
        model = torch.load('similarities/model1.pkl')
        model.cuda()
        train_loader, dev_loader, test_loader = get_dataloader(train_path, eval_path, test_path, args.cache_path, tokenizer, args.max_length, False)
        find_key_utterance('Train', model, train_loader, 3)
        find_key_utterance('Dev', model, dev_loader, 3)
        find_key_utterance('Test', model, test_loader, 3)
    else:
        model = KeyModel(config)
        print('loading...')
        train_loader, dev_loader, test_loader = get_dataloader(train_path, eval_path, test_path, args.cache_path, tokenizer, args.max_length, True)
        print('load...')
        model.cuda()
        train(model, train_loader, dev_loader, test_loader)


if __name__ == '__main__':
    main()
