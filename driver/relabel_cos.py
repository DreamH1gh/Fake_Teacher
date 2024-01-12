from pickletools import optimize
import os
import sys
sys.path.extend(["../../","../","./"])
from model.MlpModel import *
from data.Dataloader import *
from driver.Config import *
from model.BertModel import *

import numpy as np
import torch
import torch.utils.data as Data
import random
import argparse

torch.manual_seed(888)
torch.cuda.manual_seed(888)
random.seed(888)
np.random.seed(888)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='config.ptb.cfg')
argparser.add_argument('--thread', default=1, type=int, help='thread num')
argparser.add_argument('--model_name', default='mbert', type=str, help='bert, mbert, xlmr, mwordembedding')
argparser.add_argument('--source_language', default='en', type=str, help='always en in this experiment')
argparser.add_argument('--target_language', default='pl', type=str, help='de, nl, sv, fr, es, it, sk, pl, pt, he')
argparser.add_argument('--gpu', default=7, type=int, help='Use id of gpu, -1 if cpu.')

args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)
torch.set_num_threads(args.thread)
model_name = args.model_name
source_lan = args.source_language
target_lan = args.target_language

gpu = torch.cuda.is_available()
print(torch.__version__)
print("GPU available: ", gpu)
print("CuDNN: \n", torch.backends.cudnn.enabled)
config.use_cuda = False
gpu_id = 0
if (gpu and args.gpu) >= 0:
    torch.cuda.set_device(args.gpu)
    config.use_cuda = True
    print("GPU ID: ", args.gpu)
    gpu_id = args.gpu

args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)
torch.set_num_threads(args.thread)

model = MLPregressionRel()
model = model.cuda()


model_save_path = 'cross_lingual_dataset/' + target_lan + '/model_cos_head_rel'
model.load_state_dict(torch.load(model_save_path))

############################################################################################################  MLP 2.predict_step


outputFile = 'cross_lingual_dataset/' + target_lan + '/weight-' + source_lan + '-' + target_lan + '-tt.tgt.conllu.relabel_iteration_head_rel-0.1-0.6-0.2-earlystop_parallel_new_clean'
# outputFile = 'MLP_RESULT/weight_train/1536-200-200-1.conllu'

vocab = creatVocab(config.train_file, config.mbert_vocab_file, model_name)
bert = MBertExtractor(config)
bert.cuda()

data = read_weight_corpus(config.tt_vocab_file, vocab, config.max_train_len)

test_data = read_weight_corpus(config.relabel_file, vocab)

# test_data = read_weight_corpus(config.train_file_weight, vocab)

def dump_bert(bert_indices, bert_segments, bert_pieces):
    bert_outputs = bert(bert_indices, bert_segments, bert_pieces)
    return bert_outputs


output = open(outputFile, 'w', encoding='utf-8')
for i in range(len(test_data)):
    onebatch = [test_data[i]]
    bert_inputs, tags, heads, rels, lengths, masks = \
        batch_data_variable(onebatch, vocab)

    bert_indices, bert_segments, bert_pieces = bert_inputs[0].cuda(), bert_inputs[1].cuda(), bert_inputs[2].cuda()

    bert_outputs = dump_bert(bert_indices, bert_segments, bert_pieces)
    confidence = []
    for j in range(1,len(test_data[i])):
        relid = torch.from_numpy(np.array([vocab.rel2id(test_data[i][j].rel)]))
        relid = relid.cuda()
        embedding = torch.cat((bert_outputs[0][test_data[i][j].id], bert_outputs[0][test_data[i][j].head], relid)).unsqueeze(0).cuda()
        model.eval()
        pre_y = model(embedding)

        if data[i][j].weight != 0.0:
            if pre_y[0]<0:
                confidence.append(0.0000000000000001)
            else:
                confidence.append(float(pre_y[0]))
        else:
            confidence.append(0.0)
    for tree in batch_variable_depTree(onebatch, heads, rels, lengths, vocab):
        printDepTreeWeight(output, tree, confidence)

output.close()


# outputFile1 = 'cross_lingual_dataset/' + target_lan + '/weight-align-' + source_lan + '-' + target_lan + '-clean123.tgt.conllu.relabel_cos_0_head_rel'


# output1 = open(outputFile1, 'w', encoding='utf-8')
# for i in range(len(test_data)):
#     onebatch = [test_data[i]]
#     bert_inputs, tags, heads, rels, lengths, masks = \
#         batch_data_variable(onebatch, vocab)

#     bert_indices, bert_segments, bert_pieces = bert_inputs[0].cuda(), bert_inputs[1].cuda(), bert_inputs[2].cuda()

#     bert_outputs = dump_bert(bert_indices, bert_segments, bert_pieces)
#     confidence = []
#     for j in range(1,len(test_data[i])):
#         relid = torch.from_numpy(np.array([vocab.rel2id(test_data[i][j].rel)]))
#         relid = relid.cuda()
#         embedding = torch.cat((bert_outputs[0][test_data[i][j].id], bert_outputs[0][test_data[i][j].head], relid)).unsqueeze(0).cuda()
#         model.eval()
#         pre_y = model(embedding)


#         if pre_y[0]<0:
#             confidence.append(0.0000000000000001)
#         elif test_data[i][j].weight == 0.0:
#             confidence.append(0.0)
#         else:
#             confidence.append(float(pre_y[0]))
#     for tree in batch_variable_depTree(onebatch, heads, rels, lengths, vocab):
#         printDepTreeWeight(output1, tree, confidence)

# output1.close()
