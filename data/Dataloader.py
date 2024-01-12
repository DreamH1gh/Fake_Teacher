from data.Vocab import *
from collections import Counter
from data.Dependency import *
import io
import numpy as np
import torch
from torch.autograd import Variable

#读取语料库并只载入[0, max_length]长度的句子
def read_corpus(file_path, vocab=None, max_length=-1):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile, vocab):
            if max_length >= 0 and len(sentence) > max_length: continue
            data.append(sentence)
    return data

def read_weight_corpus(file_path, vocab=None, max_length=-1):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readWeightDepTree(infile, vocab):
            if max_length >= 0 and len(sentence) > max_length: continue
            data.append(sentence)
    return data

#以训练集为语料库构建vocab，统计tag、rel次数
def creatVocab(corpusFile, vocab_file, model_name):
    tag_counter = Counter()
    rel_counter = Counter()
    root = ''
    with open(corpusFile, 'r') as infile:
        for sentence in readDepTree(infile):
            for dep in sentence:
                tag_counter[dep.tag] += 1
                if dep.head != 0:
                    rel_counter[dep.rel] += 1
                elif root == '':
                    root = dep.rel
                    rel_counter[dep.rel] += 1
                elif root != dep.rel:
                    print('root = ' + root + ', rel for root = ' + dep.rel)
    if model_name == "bert" or model_name == "mbert":
        return Vocab(vocab_file, tag_counter, rel_counter, model_name, root)
    elif model_name == "xlmr":
        return Vocab(vocab_file, tag_counter, rel_counter, model_name, root)
    elif model_name == "mwordembedding":
        return Vocab(vocab_file, tag_counter, rel_counter, model_name, root)
    

#对batch内每一个句子进行操作
def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

#根据训练集生成词表对单个句子的tag(词性)和rel(关系)进行匹配编号
def sentence2id(sentence, vocab):
    result, words = [], []
    for dep in sentence:
        if not dep.pseudo: words.append(dep.form)
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        relid = vocab.rel2id(dep.rel)
        result.append([tagid, head, relid])

    return result, words

#对batch内每一个句子进行操作
def weight_sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield weight_sentence2id(sentence, vocab)

#根据训练集生成词表对单个句子的tag(词性)和rel(关系)进行匹配编号
def weight_sentence2id(sentence, vocab):
    result, words = [], []
    for dep in sentence:
        if not dep.pseudo: words.append(dep.form)
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        relid = vocab.rel2id(dep.rel)
        weight = dep.weight
        result.append([tagid, head, relid, weight])

    return result, words

#对数据集进行长度为batch的切片
def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences

#构建数据迭代器
def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

#对batch内句子的bert分词结果、tags, heads, rels, lengths, masks进行整理
def batch_data_variable(batch, vocab):
    batch_size = len(batch)
    lengths = [len(batch[b]) for b in range(batch_size)]
    max_length = lengths[0]
    for b in range(1, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]

    bert_token_indices, bert_segments_ids, bert_piece_ids = [], [], []

    #以batch内最长句子长度构建tags和masks矩阵：batch_size*max_length
    tags = Variable(torch.LongTensor(batch_size, max_length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, max_length).zero_(), requires_grad=False)
    heads = []
    rels = []
    bert_lengths = []

    b, max_bert_length = 0, 0
    for sentence, words in sentences_numberize(batch, vocab):
        # bert
        bert_indice, segments_id, piece_id = vocab.bert_ids(' '.join(words), len(words))
        cur_length = len(bert_indice)
        bert_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length
        bert_token_indices.append(bert_indice)
        bert_segments_ids.append(segments_id)
        bert_piece_ids.append(piece_id)

        # parser
        head = np.full((max_length), -1, dtype=np.int32)
        rel = np.full((max_length), -1, dtype=np.int32)
        for index, dep in enumerate(sentence):
            tags[b, index] = dep[0]
            head[index] = dep[1]
            rel[index] = dep[2]
            masks[b, index] = 1
        b += 1
        heads.append(head)
        rels.append(rel)

    bert_indices = Variable(torch.LongTensor(batch_size, max_bert_length).zero_(), requires_grad=False)
    bert_segments = Variable(torch.LongTensor(batch_size, max_bert_length).zero_(), requires_grad=False)
    bert_pieces = Variable(torch.Tensor(batch_size, max_length, max_bert_length).zero_(), requires_grad=False)

    shift_pos = 0  # start corresponds to first root
    for b in range(batch_size):
        for index in range(bert_lengths[b]):
            bert_indices[b, index] = bert_token_indices[b][index]
            bert_segments[b, index] = bert_segments_ids[b][index]

        for sindex in range(lengths[b]):
            avg_score = 1.0 / len(bert_piece_ids[b][sindex+shift_pos])
            for tindex in bert_piece_ids[b][sindex+shift_pos]:
                bert_pieces[b, sindex, tindex] = avg_score

    bert_inputs = (bert_indices, bert_segments, bert_pieces)

    return bert_inputs, tags, heads, rels, lengths, masks

def batch_weightdata_variable(batch, vocab):
    batch_size = len(batch)
    lengths = [len(batch[b]) for b in range(batch_size)]
    max_length = lengths[0]
    for b in range(1, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]

    bert_token_indices, bert_segments_ids, bert_piece_ids = [], [], []

    #以batch内最长句子长度构建tags和masks矩阵：batch_size*max_length
    tags = Variable(torch.LongTensor(batch_size, max_length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, max_length).zero_(), requires_grad=False)
    heads = []
    rels = []
    weights = []
    bert_lengths = []

    b, max_bert_length = 0, 0
    for sentence, words in weight_sentences_numberize(batch, vocab):
        # bert
        bert_indice, segments_id, piece_id = vocab.bert_ids(' '.join(words), len(words))
        cur_length = len(bert_indice)
        bert_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length
        bert_token_indices.append(bert_indice)
        bert_segments_ids.append(segments_id)
        bert_piece_ids.append(piece_id)

        # parser
        head = np.full((max_length), -1, dtype=np.int32)
        rel = np.full((max_length), -1, dtype=np.int32)
        weight = np.zeros((max_length), dtype=np.float32)
        for index, dep in enumerate(sentence):
            tags[b, index] = dep[0]
            head[index] = dep[1]
            rel[index] = dep[2]
            weight[index] = dep[3]
            masks[b, index] = 1
        b += 1
        heads.append(head)
        rels.append(rel)
        weights.append(weight)

    bert_indices = Variable(torch.LongTensor(batch_size, max_bert_length).zero_(), requires_grad=False)
    bert_segments = Variable(torch.LongTensor(batch_size, max_bert_length).zero_(), requires_grad=False)
    bert_pieces = Variable(torch.Tensor(batch_size, max_length, max_bert_length).zero_(), requires_grad=False)

    shift_pos = 0  # start corresponds to first root
    for b in range(batch_size):
        for index in range(bert_lengths[b]):
            bert_indices[b, index] = bert_token_indices[b][index]
            bert_segments[b, index] = bert_segments_ids[b][index]

        for sindex in range(lengths[b]):
            avg_score = 1.0 / len(bert_piece_ids[b][sindex+shift_pos])
            for tindex in bert_piece_ids[b][sindex+shift_pos]:
                bert_pieces[b, sindex, tindex] = avg_score

    bert_inputs = (bert_indices, bert_segments, bert_pieces)

    return bert_inputs, tags, heads, rels, lengths, masks, weights

#对batch内句子的word embedding分词结果、tags, heads, rels, lengths进行整理
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def batch_data_wordembedding(batch, vocab, language):
    batch_size = len(batch)
    lengths = [len(batch[b]) for b in range(batch_size)]
    max_length = lengths[0]
    for b in range(1, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]


    path = "MultilingualModle/my-mwordembedding/wiki.multi." + language + ".txt"

    src_embeddings, src_id2word, src_word2id = load_vec(path, 50000)
    #以batch内最长句子长度构建tags和masks矩阵：batch_size*max_length
    tags = Variable(torch.LongTensor(batch_size, max_length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, max_length).zero_(), requires_grad=False)
    heads = []
    rels = []
    sentence_lengths, batch_embedding = [], []

    b, max_bert_length = 0, 0
    for sentence, words in sentences_numberize(batch, vocab):
        sentence_embedding = []
        for i in range(len(words)):
            try:
                sentence_embedding.append(torch.tensor(src_embeddings[src_word2id[words[i]]]))
            except:
                sentence_embedding.append(torch.tensor(np.zeros((300,), dtype=float)))

        cur_length = len(sentence_embedding)
        sentence_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length

        # parser
        head = np.full((max_length), -1, dtype=np.int32)
        rel = np.full((max_length), -1, dtype=np.int32)
        for index, dep in enumerate(sentence):
            tags[b, index] = dep[0]
            head[index] = dep[1]
            rel[index] = dep[2]
            masks[b, index] = 1
        b += 1
        heads.append(head)
        rels.append(rel)
        batch_embedding.append(sentence_embedding)

    embeddings = Variable(torch.FloatTensor(batch_size, max_bert_length + 1, 300).zero_(), requires_grad=False)

    for b in range(batch_size):
        for index in range(sentence_lengths[b]):
            embeddings[b, index + 1] = batch_embedding[b][index]


    return embeddings, tags, heads, rels, lengths, masks

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx])))
        yield sentence
