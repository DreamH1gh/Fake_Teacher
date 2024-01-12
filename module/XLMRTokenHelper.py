from transformers import AutoTokenizer


class XLMRTokenHelper(object):
    def __init__(self, xlmr_tokenizer_file):
        self.tokenizer = AutoTokenizer.from_pretrained(xlmr_tokenizer_file)
        print("Load xlmr vocabulary finished")
        self.key_words = ("[UNK]", "</s>","[PAD]", "<s>", "[MASK]")

    def bert_ids(self, text, length):
        text = text.replace('##', '@@')
        text = text.replace('``', "''")
        text = text.replace('\u200b\u200b', "")
        text = text.replace('\xa0',"")
        outputs = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        bert_indice = outputs["input_ids"].squeeze(0)
        segments_id = outputs["attention_mask"].squeeze(0)

        list_bert_indice = [idx.item() for idx in bert_indice]
        list_segments_id = [idx.item() for idx in segments_id]

        bert_tokens = self.tokenizer.convert_ids_to_tokens(list_bert_indice)
        org_tokens = text.split()
        tokens = self.tokenizer.convert_tokens_to_string(bert_tokens)

        list_piece_id = []
        org_id, last_id = -1, -1
        last_word = ''
        able_continue = False
        for idx, bpe_u in enumerate(bert_tokens):
            if bpe_u in self.key_words:
                last_id += 1
                if bpe_u == '[UNK]':
                    org_id += 1
                last_word = bpe_u
                list_piece_id.append([idx])
                able_continue = False
            # elif not(bpe_u.startswith("▁")) or (able_continue and len(last_word) < len(org_tokens[org_id]) + 1):
            elif not(bpe_u.startswith("▁")):
                list_piece_id[last_id].append(idx)
                if not(bpe_u.startswith("▁")): last_word = last_word + bpe_u
                else: last_word = last_word + bpe_u
                able_continue = True
            else:
                last_id += 1
                org_id += 1
                last_word = bpe_u
                able_continue = True
                list_piece_id.append([idx])
        if org_id != length-1:
            print("An error occurs in bert tokenizer: token len not aligns")

        return list_bert_indice, list_segments_id, list_piece_id
