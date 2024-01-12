import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import random
import argparse
from driver.Config import *
from model.ParserModel import *
from model.BertModel import *
from driver.ParserHelper import *
from data.Dataloader import *
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle
import json


def train(data, dev_data, test_data, parser, vocab, config, model_name, source_lan, target_lan):
    global_step = 0
    best_UAS = 0
    best_test_uas, best_test_las = 0, 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))

    optimizers, schedulers = [], []
    optimizer_model = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)
    optimizers.append(optimizer_model)

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
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.train_iters*batch_num)
        optimizers.append(optimizer_bert)
        schedulers.append(scheduler_bert)

    all_params = [p for p in parser.model.parameters() if p.requires_grad]
    if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
        for group in optimizer_bert_parameters:
            for p in group['params']:
                all_params.append(p)

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
                bert_inputs, tags, heads, rels, lengths, masks = \
                    batch_data_variable(onebatch, vocab)
                parser.model.train()
                parser.bert.train()
                parser.forward(bert_inputs, tags, masks)
            if model_name == "mwordembedding":
                embeddings, tags, heads, rels, lengths, masks = \
                    batch_data_wordembedding(onebatch, vocab, source_lan)
                parser.model.train()
                parser.forward(embeddings, tags, masks)
                
            loss = parser.compute_loss(heads, rels, lengths)
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

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num or global_step ==2000:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step), model_name, source_lan, target_lan, "dev")
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                if model_name == 'mwordembedding' and global_step >= 1000:
                    arc_correct, rel_correct, arc_total, test_uas, test_las, pre_arcs, pre_rels = \
                        evaluate_test(test_data, parser, vocab, config.test_file + '.' + str(global_step), model_name, source_lan, target_lan, "test")
                    sys.exit(0)
                if model_name == 'xlmr' and global_step >= 2000:
                    arc_correct, rel_correct, arc_total, test_uas, test_las, pre_arcs, pre_rels = \
                        evaluate_test(test_data, parser, vocab, config.test_file + '.' + str(global_step), model_name, source_lan, target_lan, "test")
                    sys.exit(0)
                if dev_uas > best_UAS:
                    print("Exceed best uas: history = %.2f, current = %.2f" %(best_UAS, dev_uas))
                    best_UAS = dev_uas
                    # best_test_uas = test_uas
                    # best_test_las = test_las
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), save_model_path)


def evaluate(data, parser, vocab, outputFile, model_name, source_lan, target_lan, data_source):
    start = time.time()
    parser.model.eval()
    if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
        parser.bert.eval()
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
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1


    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las

def evaluate_test(data, parser, vocab, outputFile, model_name, source_lan, target_lan, data_source):
    start = time.time()
    parser.model.eval()
    if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
        parser.bert.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
    pre_arcs = ["NULL"] * len(data)
    pre_rels = ["NULL"] * len(data)
    times = 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        if model_name == "bert" or model_name == "mbert" or model_name == "xlmr":
            bert_inputs, tags, heads, rels, lengths, masks = \
                batch_data_variable(onebatch, vocab)
            arcs_batch, rels_batch, pre_arc, pre_rel = parser.outputparse(bert_inputs, tags, lengths, masks)
        elif model_name == "mwordembedding":
            if data_source == "dev":
                embeddings, tags, heads, rels, lengths, masks = \
                        batch_data_wordembedding(onebatch, vocab, source_lan)
            if data_source == "test":
                embeddings, tags, heads, rels, lengths, masks = \
                        batch_data_wordembedding(onebatch, vocab, target_lan)
            arcs_batch, rels_batch, pre_arc, pre_rel = parser.outputmweparse(embeddings, tags, lengths, masks)
        count = 0
        for i in range(len(pre_arc)):
            pre_arcs[times * config.test_batch_size + i] = pre_arc[i]
            pre_rels[times * config.test_batch_size + i] = pre_rel[i]
        times += 1
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    output2json = {
        "pre_arcs":pre_arcs,
        "pre_rels":pre_rels
    }
    filename = 'pre_result/' + source_lan + '2' + target_lan + '/' + model_name + '.json'
    with open(filename,'w') as file_obj:
        json.dump(output2json,file_obj)

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las, pre_arcs, pre_rels


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
    argparser.add_argument('--model_name', default='mwordembedding', type=str, help='bert, mbert, xlmr, mwordembedding')
    argparser.add_argument('--source_language', default='en', type=str, help='always en in this experiment')
    argparser.add_argument('--target_language', default='pl', type=str, help='de, nl, sv, fr, es, it, sk, pl, pt, he')
    argparser.add_argument('--gpu', default=5, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    model_name = args.model_name

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
    if model_name == "bert":
        save_bert_dir = "ModelResult/bert"
        save_model_path = save_bert_dir + "/model/" + language_transfer
        save_vocab_path = save_bert_dir + "/vocab"
        vocab = creatVocab(config.train_file, config.bert_vocab_file, model_name)
        pickle.dump(vocab, open(save_vocab_path, 'wb'))
        bert = BertExtractor(config)
        model = ParserModel(vocab, config, bert.bert_hidden_size)

        if config.use_cuda:
            #torch.backends.cudnn.enabled = True
            model = model.cuda()
            bert = bert.cuda()
        parser = BiaffineParser(model, bert, vocab.ROOT)

    elif model_name == "mbert":
        save_bert_dir = "ModelResult/mbert"
        save_model_path = save_bert_dir + "/model/" + language_transfer
        save_vocab_path = save_bert_dir + "/vocab"
        vocab = creatVocab(config.train_file, config.mbert_vocab_file, model_name)
        pickle.dump(vocab, open(save_vocab_path, 'wb'))
        mbert = MBertExtractor(config)
        model = ParserModel(vocab, config, mbert.bert_hidden_size)

        if config.use_cuda:
            #torch.backends.cudnn.enabled = True
            model = model.cuda()
            mbert = mbert.cuda()
        parser = BiaffineParser(model, mbert, vocab.ROOT)

    elif model_name == "xlmr":
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

    elif model_name == "mwordembedding":
        save_bert_dir = "ModelResult/mwordembedding"
        save_model_path = save_bert_dir + "/model/" + language_transfer
        save_vocab_path = save_bert_dir + "/vocab"
        vocab = creatVocab(config.train_file, "MultilingualModle/my-mwordembedding/wiki.multi.en.txt", model_name)
        pickle.dump(vocab, open(save_vocab_path, 'wb'))
        model = ParserModel(vocab, config, 300)

        if config.use_cuda:
            model = model.cuda()
        parser = MWE_BiaffineParser(model, vocab.ROOT)

    else:
        print("check the model name!")
        sys.exit (0)

    data = read_corpus(config.train_file, vocab, config.max_train_len)
    dev_data = read_corpus(config.dev_file, vocab, config.max_train_len)
    test_data = read_corpus(config.target_file, vocab, config.max_train_len)

    train(data, dev_data, test_data, parser, vocab, config, model_name, source_lan, target_lan)
