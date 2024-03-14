import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from similarity_model import get_dataloader, SModel
from sklearn.metrics import accuracy_score, recall_score
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm


def train(model, train_loader, dev_loader, test_loader):
    max_acc = 0
    max_test_acc = 0

    log_file = 'similarities/results3.txt'
    logss = open(log_file, 'w')

    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    t_total = len(train_loader) * 5
    num_warmup_steps = int(t_total * 0.1)
    # scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = t_total // (5 * 5)
    steps = 0

    for epoch in range(5):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'question_input_ids': batch['question_input_ids'],
                      'question_attention_mask': batch['question_attention_mask'],
                      'utterance_input_ids': batch['utterance_input_ids'],
                      'utterance_attention_mask': batch['utterance_attention_mask'],
                      'label': batch['label']}
            outputs = model(**inputs)
            loss = outputs[1]
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
                eval_result = evaluate('dev', model, dev_loader)
                test_result = evaluate('test', model, test_loader)
                print("Eval Result:", eval_result)
                print("Test Result:", test_result)
                if eval_result > max_acc:
                    max_acc = eval_result
                    max_test_acc = test_result
                    torch.save(model, 'similarities/model3.pkl')
                logss.write("    Eval Result: {}\n".format(eval_result))
                logss.write("    Test Result: {}\n".format(test_result))
            steps += 1
    eval_result = evaluate('dev', model, dev_loader)
    test_result = evaluate('test', model, test_loader)
    if eval_result > max_acc:
        max_acc = eval_result
        max_test_acc = test_result
        torch.save(model, 'similarities/model3.pkl')
    print("Eval Result:", eval_result)
    print("Test Result:", test_result)
    logss.write("\nFinal Eval Result: {}\n".format(max_acc))
    logss.write("Final Test Result: {}\n".format(max_test_acc))

    logss.close()


def evaluate(eval_type, model, eval_dataloader):
    model.eval()

    preds = []
    gold = []
    pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))

    with torch.no_grad():
        for _, batch in pbar:
            inputs = {'question_input_ids': batch['question_input_ids'],
                      'question_attention_mask': batch['question_attention_mask'],
                      'utterance_input_ids': batch['utterance_input_ids'],
                      'utterance_attention_mask': batch['utterance_attention_mask'],
                      'label': batch['label']}
            outputs = model(**inputs)
            logits = outputs[0]
            pred = logits.gt(0.5).long().cpu().numpy()
            preds.append(pred)
            gold.append(batch['label'].data.cpu().long().numpy())
    preds = np.concatenate(preds)
    gold = np.concatenate(gold)
    # accuracy = round(accuracy_score(gold, preds), 4)
    recall = target_recall(gold, preds)

    model.train()
    return recall


def target_recall(y_true, y_pred):
    golden_num = 0
    right_num = 0
    for i, j in zip(y_true, y_pred):
        if i == 1:
            golden_num += 1
            if j == i:
                right_num += 1
    return round(right_num / golden_num, 4)


def main():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    model = SModel()
    print('loading...')
    train_loader, dev_loader, test_loader = get_dataloader('data/friendsqa_question_context_train.json',
                                                           'data/friendsqa_question_context_dev.json',
                                                           'data/friendsqa_question_context_test.json',
                                                           model.tokenizer, 64, shuffle=True)
    print('load...')
    model.cuda()
    train(model, train_loader, dev_loader, test_loader)


if __name__ == '__main__':
    main()

