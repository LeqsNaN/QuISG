import os
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='Parameters for FriendsQA dataset')

parser.add_argument('-lr', '--learning_rate', type=float, default=4e-6)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1)
parser.add_argument('-eps', '--epochs', type=int, default=3)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='electra')
parser.add_argument('-cp', '--cache_path', type=str, default='cache1')
parser.add_argument('-ml', '--max_length', type=int, default=512)
parser.add_argument('-qml', '--question_max_length', type=int, default=32)
parser.add_argument('-bsz', '--batch_size', type=int, default=4)
parser.add_argument('-elsp', '--early_stop_patience', type=int, default=10)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-6)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='saves_graph_seed_1')
parser.add_argument('--colab', type=bool, default=False)
parser.add_argument('--pred_file', type=str, default='pred.json')
parser.add_argument('--mha_layer_num', type=int, default=3)
parser.add_argument('--model_num', type=int, default=1)
parser.add_argument('--use_cls_for_gather', type=int, default=0, help='whether to use [CLS] as the token to gather speaker/utter information')
parser.add_argument('--draw', type=int, default=0, help='whether to draw attention')
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--model_kind', type=str, default='rel_graph')
parser.add_argument('--trm_layers', type=int, default=0)
parser.add_argument('--ku_layers', type=int, default=3)
parser.add_argument('--nfeat', type=int, default=1024)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nheads', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--gat_dropout', type=float, default=0.1)
parser.add_argument('--gat_num_layer', type=int, default=2)
parser.add_argument('--context_range', type=int, default=2)
parser.add_argument('--n_gram', type=int, default=3)
parser.add_argument('--eval_key_utterance', action='store_true')
parser.add_argument('--cls_type', type=str, default='wcontext')
parser.add_argument('--eval_speaker_cls', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--lower_weight', type=float, default=0.3)
parser.add_argument('--higher_weight', type=float, default=0.7)
parser.add_argument('--num_node_type', type=int, default=7)
parser.add_argument('--num_rel', type=int, default=3)
parser.add_argument('--add_utter_mask', action='store_true')
parser.add_argument('--use_all_context', action='store_true')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--hard_rate', type=float, default=0.1)
parser.add_argument('--eval_hard_rate', type=float, default=0.1)
parser.add_argument('--hard_before_softmax', action='store_true')
parser.add_argument('--type_ab', type=str, default='key')
parser.add_argument('--topp', type=int, default=2)
parser.add_argument('--simcse_open', action='store_true')

args = parser.parse_args()

if not os.path.exists('saves_graph_seed_1' if not args.small else 'saves_small'):
    os.mkdir('saves_graph_seed_1' if not args.small else 'saves_small')
if not os.path.exists('caches_graph' if not args.small else 'caches_small'):
    os.mkdir('caches_graph' if not args.small else 'caches_small')

args.add_speaker_mask = args.model_num

args.save_path = ('saves_graph_seed_1/' if not args.small else 'saves_small/') + args.model_type + '_' + args.save_path
args.cache_path = ('caches_graph/' if not args.small else 'caches_small/') + args.model_type + '_' + args.cache_path
args.model_name = '/mnt/sdb/ljn/pretrained/electra-large-discriminator'

args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + args.model_kind + '_' + str(args.max_length) + '_seed_' + str(args.seed) + '_index_' + str(args.index)
# args.save_path += '_' + args.type_ab + '_' + str(args.max_length) + '_seed_' + str(args.seed) + '_index_' + str(args.index)
if args.use_cls_for_gather:
    args.cache_path += '_cls'
    args.save_path += '_cls'
args.pred_file = args.save_path + '/' + args.pred_file
