"""
简单分词器实现，用于替代transformers库的AutoTokenizer
"""

import re
from collections import Counter
import torch


class SimpleVocab:
    def __init__(self):
        # 添加特殊token
        self.word2idx = {
            '<PAD>': 0,    # 填充token
            '<UNK>': 1,    # 未知token
            '<CLS>': 2,    # 分类token
            '<SEP>': 3     # 分隔符token
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def add_words(self, words):
        """批量添加单词到词汇表"""
        for word in words:
            self.add_word(word)

    def __len__(self):
        return self.vocab_size

    def build_from_texts(self, texts):
        """从文本构建词汇表"""
        word_counts = Counter()
        for text in texts:
            words = text.split()
            # 进一步处理标点符号
            processed_words = []
            for word in words:
                # 分离常见标点符号
                word = re.sub(r'([^\w\s])', r' \1 ', word) # 前后加空格
                processed_words.extend(word.split())
            word_counts.update(processed_words)
        
        # 添加高频词到词汇表（出现次数>=2的词）
        for word, count in word_counts.items():
            if count >= 2:
                self.add_word(word)
        
        return self


class SimpleTokenizer:
    """简单分词器"""
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, text, max_length=128):
        """对文本进行分词"""
        # 分离标点符号（文本已在data_processing中转为小写）
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        words = text.split()

        tokens = ['<CLS>'] + words
        
        # 统一长度
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend(['<PAD>'] * (max_length - len(tokens)))
        
        return tokens

    def encode(self, text, max_length=128):
        """将文本编码为ID序列"""
        tokens = self.tokenize(text, max_length)
        input_ids = [self.vocab.word2idx.get(token, self.vocab.word2idx['<UNK>']) for token in tokens]
        # 注意力掩码，非<PAD>为1，<PAD>为0
        attention_mask = [1 if token != '<PAD>' else 0 for token in tokens]
        return input_ids, attention_mask

    def __call__(self, text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'):
        """模仿transformers tokenizer接口"""
        input_ids, attention_mask = self.encode(text, max_length)
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }