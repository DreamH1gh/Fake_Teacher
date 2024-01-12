from module.BertTokenHelper import *
from module.XLMRTokenHelper import *

class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2

    def __init__(self, vocab_file, tag_counter, rel_counter, model_name, relroot='root'):
        self._root = relroot
        self._id2tag = ['<pad>', relroot, '<unk>']
        self._id2rel = ['<pad>', relroot, '<unk>']
        self.model_name = model_name

        for tag, count in tag_counter.most_common():
            if tag != relroot: self._id2tag.append(tag)

        for rel, count in rel_counter.most_common():
            if rel != relroot: self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))

        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: POS tags dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #tags %d, #rels %d" % (self.tag_size, self.rel_size))
        
        if model_name == "bert" or model_name == "mbert":
            self.tokenizer = BertTokenHelper(vocab_file)
        if model_name == "xlmr":
            self.tokenizer = XLMRTokenHelper(vocab_file)

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id.get(x, self.UNK) for x in xs]
        return self._rel2id.get(xs, self.UNK)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    def bert_ids(self, text, length):
        if self.model_name == "bert" or self.model_name == "mbert" or self.model_name == "xlmr":
            outputs = self.tokenizer.bert_ids(text, length)
            return outputs

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)
