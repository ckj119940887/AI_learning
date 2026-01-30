import torch
import torch.nn as nn
import jieba

text = "自然语言是由文字构成的，而语言的含义是由单词构成的。即单词是含义的最小单位。因此为了让计算机理解自然语言，首先要让它理解单词含义。"


# call jieba to split
original_words = jieba.lcut(text, cut_all=False)
print(original_words)

# filter word
stopwords = {"的", "是", "而", "由", "，", "。"}
words = [word for word in original_words if word not in stopwords]
print(words)

# construct id2word
# delete repeated word
vocab = list(set(words))
print(vocab)
print(len(vocab))

# construct word2id
word2id = dict()
for id, word in enumerate(vocab):
    word2id[word] = id
print(word2id)

# define embedding
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=5)

for id, word in enumerate(vocab):
    word_vec = embedding(torch.tensor(id))
    print(f"id: {id}, word: {word}, vec: {word_vec.detach().numpy()}")