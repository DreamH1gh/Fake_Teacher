import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import random
import argparse
import copy
from driver.Config import *
from model.MlpModel import *
from model.ParserModel import *
from model.BertModel import *
from driver.ParserHelper import *
from data.Dataloader import *
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle


def train(data, dev_data, test_data, parser, vocab, config, model_name, source_lan, target_lan, relabel_type):
    global_step = 0
    best_UAS = 0
    best_test_uas, best_test_las = 0, 0
    early_stop = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    relabel_data = copy.deepcopy(test_data)

    optimizers, schedulers = [], []
    optimizer_model = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)
    optimizers.append(optimizer_model)

    mlp_model = MLPregressionRel()
    mlp_model = mlp_model.cuda()
    model_read_path = 'cross_lingual_dataset/' + target_lan + '/model_cos_head_rel'
    mlp_model.load_state_dict(torch.load(model_read_path))

    no_decay = ['bias', 'LayerNorm.weight']
    if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
        optimizer_bert_parameters = [
            {'params': [p for n, p in parser.bert.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad ],
            'weight_decay': config.decay},
            {'params': [p for n, p in parser.bert.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0}
        ]
        optimizer_bert = AdamW(optimizer_bert_parameters, lr=5e-5, eps=1e-8)
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=100*batch_num)
        optimizers.append(optimizer_bert)
        schedulers.append(scheduler_bert)

    all_params = [p for p in parser.model.parameters() if p.requires_grad]
    if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
        for group in optimizer_bert_parameters:
            for p in group['params']:
                all_params.append(p)

    for iter in range(1000):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        data_iteration = copy.deepcopy(data)
        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch in data_iter(data_iteration, config.train_batch_size, True):
            if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
                bert_inputs, tags, heads, rels, lengths, masks, weights = \
                    batch_weightdata_variable(onebatch, vocab)
                parser.model.train()
                parser.bert.train()
                parser.forward(bert_inputs, tags, masks)
            if model_name == "mwordembedding":
                embeddings, tags, heads, rels, lengths, masks = \
                    batch_data_wordembedding(onebatch, vocab, source_lan)
                parser.model.train()
                parser.forward(embeddings, tags, masks)
                
            # loss = parser.compute_loss(heads, rels, lengths)
            loss = parser.compute_weight_loss(heads, rels, lengths, weights)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            arc_correct, label_correct, total_arcs = parser.compute_accuracy(heads, rels)
            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = overall_arc_correct * 100.0 / overall_total_arcs
            las = overall_label_correct * 100.0 / overall_total_arcs
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, loss:%.2f" \
                %(global_step, uas, las, iter, batch_iter, overall_total_arcs, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(all_params, max_norm=config.clip)
                for optimizer in optimizers:
                    optimizer.step()
                if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
                    for scheduler in schedulers:
                        scheduler.step()
                    parser.bert.zero_grad()
                parser.model.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step), model_name, source_lan, target_lan, "dev")
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                early_stop += 1
                if dev_uas > best_UAS:
                    print("Exceed best uas: history = %.2f, current = %.2f" %(best_UAS, dev_uas))
                    best_UAS = dev_uas
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), save_model_path)
                    early_stop = 0
                print(early_stop)
                print("Best Dev Test: uas %.2f, las %.2f" % \
                      (best_test_uas, best_test_las))
    
        if (iter + 1) % 8 == 0:
        # if global_step % 500 == 0:
            if relabel_type == 'normal':
                relabel_data = relabel(test_data, parser, vocab, relabel_data)
            if relabel_type == 'iteration':
                relabel_data, data = relabel_iteration(test_data, parser, vocab, relabel_data, mlp_model, data)
        if early_stop > 25:
            if relabel_type == 'normal':
                outputFile = config.tt_weight_align_train_file + '.relabel-0.1-0.6-0.2-earlystop_parallel'
            if relabel_type == 'iteration':
                outputFile = config.tt_weight_align_train_file + '.relabel_iteration_head_rel-0.1-0.6-0.2-earlystop_parallel_new'
            output = open(outputFile, 'w', encoding='utf-8')
            for i in range(len(relabel_data)):
                for j in range(1, len(relabel_data[i])):
                    values = [str(relabel_data[i][j].id), relabel_data[i][j].org_form, "_", relabel_data[i][j].tag, \
                            relabel_data[i][j].tag, "_", str(relabel_data[i][j].head), relabel_data[i][j].rel, "_", "_", str(relabel_data[i][j].weight)]
                    output.write('\t'.join(values) + '\n')
                output.write('\n')
            output.close()
            sys.exit(0)

        # if iter == (config.train_iters - 1):
        #     start = time.time()
        #     parser.model.eval()
        #     relabel_data = copy.deepcopy(test_data)
        #     num = 0
        #     outputFile = config.tt_weight_align_train_file + '.relabel'
        #     output = open(outputFile, 'w', encoding='utf-8')
        #     for onebatch in data_iter(test_data, config.test_batch_size, False):
        #         bert_inputs, tags, heads, rels, lengths, masks = \
        #             batch_data_variable(onebatch, vocab)
        #         arcs_batch, rels_batch = parser.parse(bert_inputs, tags, lengths, masks)
        #         count = 0
                
        #         for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
        #             # printDepTree(output, tree)
        #             for j in range(1, len(onebatch[count])):
        #                 if onebatch[count][j].weight == 0.0:
        #                     relabel_data[num * config.test_batch_size + count][j].head = tree[j].head
        #                     relabel_data[num * config.test_batch_size + count][j].rel = tree[j].rel
        #                 values = [str(relabel_data[num * config.test_batch_size + count][j].id), relabel_data[num * config.test_batch_size + count][j].org_form, "_", relabel_data[num * config.test_batch_size + count][j].tag, \
        #                         relabel_data[num * config.test_batch_size + count][j].tag, "_", str(relabel_data[num * config.test_batch_size + count][j].head), relabel_data[num * config.test_batch_size + count][j].rel, "_", "_", relabel_data[num * config.test_batch_size + count][j].weight]
        #                 output.write('\t'.join(values) + '\n')
        #             count += 1
        #         num += 1
        #     output.close()
            
def relabel(data, parser, vocab, relabel_data):
    start = time.time()
    parser.model.eval()
    num = 0
    for onebatch in data_iter(data, config.test_batch_size, False):
        bert_inputs, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, vocab)
        arcs_batch, rels_batch = parser.parse(bert_inputs, tags, lengths, masks)
        count = 0
        
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            # printDepTree(output, tree)
            for j in range(len(onebatch[count])):
                if onebatch[count][j].weight == 0.0:
                    relabel_data[num * config.test_batch_size + count][j].head = tree[j].head
                    relabel_data[num * config.test_batch_size + count][j].rel = tree[j].rel
            count += 1
        num += 1
    return relabel_data

def relabel_iteration(test_data, parser, vocab, relabel_data, mlp_model, data):
    start = time.time()
    parser.model.eval()
    mlp_model.eval()

    mbert_vocab = creatVocab(config.train_file, config.mbert_vocab_file, 'mbert')
    mbert = MBertExtractor(config)
    mbert.cuda()
    relabel_weight = []
    num = 0
    over_size_num = 0
    for onebatch in data_iter(test_data, config.test_batch_size, False):
        bert_inputs, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, vocab)

        mbert_inputs, _, _, _, _, _ = \
            batch_data_variable(onebatch, mbert_vocab)
        mbert_indices, mbert_segments, mbert_pieces = mbert_inputs[0].cuda(), mbert_inputs[1].cuda(), mbert_inputs[2].cuda()
        mbert_outputs = mbert(mbert_indices, mbert_segments, mbert_pieces)

        arcs_batch, rels_batch = parser.parse(bert_inputs, tags, lengths, masks)
        count = 0
        relabel_num = 0

        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            # printDepTree(output, tree)
            for j in range(len(onebatch[count])):
                if relabel_data[num * config.test_batch_size + count][j].weight == 0.0:
                    if align_pro[num * config.test_batch_size + count + over_size_num].shape[0] > config.max_train_len + 1:
                        over_size_num += 1

                    if align_pro[num * config.test_batch_size + count + over_size_num][j][tree[j].head][vocab.rel2id(tree[j].rel)] > 0.1:
                        relabel_data[num * config.test_batch_size + count][j].head = tree[j].head
                        relabel_data[num * config.test_batch_size + count][j].rel = tree[j].rel
                        relid = torch.from_numpy(np.array([vocab.rel2id(tree[j].rel)]))
                        relid = relid.cuda()
                        embedding = torch.cat((mbert_outputs[count][tree[j].id], mbert_outputs[count][tree[j].head], relid)).unsqueeze(0).cuda()
                        pre_y = mlp_model(embedding)
                        if pre_y < 0:
                            pre_y = 0.0000000000000001
                            relabel_weight.append(pre_y)
                            relabel_data[num * config.test_batch_size + count][j].weight = pre_y
                        else:
                            relabel_weight.append(pre_y.item())
                            relabel_data[num * config.test_batch_size + count][j].weight = pre_y.item()
                        relabel_num += 1
            count += 1
        num += 1
    print(relabel_num)
    # if relabel_num == 0:
    #     if relabel_type == 'normal':
    #         outputFile = config.tt_weight_align_train_file + '.relabel-0.1-0.7-0.2-nostop'
    #     if relabel_type == 'iteration':
    #         outputFile = config.tt_weight_align_train_file + '.relabel_iteration_head_rel-0.1-0.7-0.2-nostop'
    #     output = open(outputFile, 'w', encoding='utf-8')
    #     for i in range(len(relabel_data)):
    #         for j in range(1, len(relabel_data[i])):
    #             values = [str(relabel_data[i][j].id), relabel_data[i][j].org_form, "_", relabel_data[i][j].tag, \
    #                     relabel_data[i][j].tag, "_", str(relabel_data[i][j].head), relabel_data[i][j].rel, "_", "_", str(relabel_data[i][j].weight)]
    #             output.write('\t'.join(values) + '\n')
    #         output.write('\n')
    #     output.close() 
    #     sys.exit(0)

    relabel_count = 0
    relabel_weight.sort()
    relabel_threshold = relabel_weight[int(len(relabel_weight) * 0.8)]
    for i in range(len(relabel_data)):
        for j in range(len(relabel_data[i])):
            if relabel_data[i][j].weight >= relabel_threshold and relabel_data[i][j].weight > 0.0000000000000001:
                data[i][j].head = relabel_data[i][j].head
                data[i][j].rel = relabel_data[i][j].rel
                data[i][j].weight = 1.0
                relabel_count += 1
            else:
                relabel_data[i][j].weight = 0.0

    return relabel_data, data


def evaluate(data, parser, vocab, outputFile, model_name, source_lan, target_lan, data_source):
    start = time.time()
    parser.model.eval()
    # parser.bert.eval()
    # output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
            bert_inputs, tags, heads, rels, lengths, masks = \
                batch_data_variable(onebatch, vocab)
            arcs_batch, rels_batch = parser.parse(bert_inputs, tags, lengths, masks)
        elif model_name == "mwordembedding":
            if data_source == "dev":
                embeddings, tags, heads, rels, lengths, masks = \
                        batch_data_wordembedding(onebatch, vocab, source_lan)
            if data_source == "test":
                embeddings, tags, heads, rels, lengths, masks = \
                        batch_data_wordembedding(onebatch, vocab, target_lan)
            arcs_batch, rels_batch = parser.parse(embeddings, tags, lengths, masks)
        count = 0
        
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            # printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    # output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    torch.manual_seed(888)
    torch.cuda.manual_seed(888)
    random.seed(888)
    np.random.seed(888)

    # gpu
    gpu = torch.cuda.is_available()
    print(torch.__version__)
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config.ptb.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--model_name', default='xlmr', type=str, help='bert, mbert, xlmr, mwordembedding')
    argparser.add_argument('--source_language', default='en', type=str, help='always en in this experiment')
    argparser.add_argument('--target_language', default='he', type=str, help='de, nl, sv, fr, es, it, sk, pl, pt, he')
    argparser.add_argument('--relabel_type', default='iteration', type=str, help='normal, iteration')
    argparser.add_argument('--gpu', default=5, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    model_name = args.model_name
    relabel_type = args.relabel_type


    config.use_cuda = False
    gpu_id = 0
    if (gpu and args.gpu) >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)
        gpu_id = args.gpu


    source_lan = args.source_language
    target_lan = args.target_language
    language_transfer = source_lan + "2" + target_lan

    save_bert_dir = "ModelResult/xlmr"
    save_model_path = save_bert_dir + "/model/" + language_transfer
    save_vocab_path = save_bert_dir + "/vocab"
    vocab = creatVocab(config.train_file, config.xlmr_tokenizer_file, model_name)
    pickle.dump(vocab, open(save_vocab_path, 'wb'))
    xlmr = XLMRExtractor(config)
    model = ParserModel(vocab, config, xlmr.bert_hidden_size)

    if config.use_cuda:
        #torch.backends.cudnn.enabled = True
        model = model.cuda()
        xlmr = xlmr.cuda()
    parser = BiaffineParser(model, xlmr, vocab.ROOT)


    count = 0
    threshold_weight = []
    data = read_weight_corpus(config.tt_weight_cos_train_file, vocab, config.max_train_len)
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            threshold_weight.append(data[i][j].weight)
    threshold_weight.sort()
    threshold = threshold_weight[int(len(threshold_weight) * 0.4)]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j].weight >= threshold:
                data[i][j].weight = 1.0
            else:
                data[i][j].weight = 0.0
                count += 1
    
    dev_data = read_corpus(config.dev_file, vocab, config.max_train_len)
    test_data = read_weight_corpus(config.tt_weight_cos_train_file, vocab, config.max_train_len)
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            if test_data[i][j].weight >= threshold:
                continue
            else:
                test_data[i][j].weight = 0.0

train_path = 'cross_lingual_dataset/' + target_lan + '/align_weight_0.1_parallel.npy'
with open(train_path, 'rb') as f:
    align_pro = np.load(f, allow_pickle=True)


    train(data, dev_data, test_data, parser, vocab, config, model_name, source_lan, target_lan, relabel_type)
