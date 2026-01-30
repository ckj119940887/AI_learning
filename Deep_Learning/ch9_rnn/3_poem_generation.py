import torch
import torch.nn as nn
import torch.optim as optim
from sympy import sequence
from torch.utils.data import Dataset, DataLoader
import re

def preprocess(file_path):
    poem_list = []
    char_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除标点符号
            line = re.sub(r'[，。、？！：]', '', line).strip()
            #
            poem_list.append(list(line))
            # 按字去重
            char_set.update(list(line))

    # 构建词表
    vocab = list(char_set) + ["<UNK>"]
    word2id = {word:id for id, word in enumerate(vocab)}

    # convert poem to id list
    id_sequences = []
    for poem in poem_list:
        id_seq = [word2id.get(word) for word in poem]
        id_sequences.append(id_seq)

    return id_sequences, vocab, word2id

id_sequences, vocab, word2id = preprocess("../data/poems.txt")
print(len(id_sequences))
print(len(vocab))
print(len(word2id))

# define dataset
class PoetryDataset(Dataset):
    def __init__(self, id_sequences, seq_len):
        self.data = []
        self.seq_len = seq_len
        # iterate over id_sequences, 截取长度为L的序列x和后续序列y
        for seq in id_sequences:
            # iterate over every word
            for i in range(0, len(seq)-self.seq_len):
                self.data.append((seq[i:i+self.seq_len], seq[i+1:i+self.seq_len+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y

dataset = PoetryDataset(id_sequences, 24)
print(len(dataset))

# define model
class PoetryRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hx=None):
        embedded = self.embedding(input)
        output, hn = self.rnn(embedded, hx)
        output = self.linear(output)
        return output, hn

model = PoetryRNN(len(vocab), embedding_dim=256, hidden_size=512, num_layers=2)

# train model
def train(model, dataset, lr, epoch_num, batch_size, device):
    model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        loss_total = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss_value = loss(output.transpose(1, 2), y)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_total += loss_value.item() * x.shape[0]
            print(f"\repoch:{epoch:0>2}[{'=' * (int((batch_idx + 1) / len(dataloader) * 50)):<50}]", end="")


        print(f"loss: {loss_total / len(dataset):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, dataset, lr=1e-3, epoch_num=20, batch_size=32, device=device)

# generate text
def generate(model, vocab, word2id, start_token, line_num=4, line_length=7):
    model.eval()
    poem = []
    current_len = line_length # 保存每行剩余字数
    start_token_id = word2id.get(start_token, word2id['<UNK>'])
    if start_token_id != word2id['<UNK>']:
        poem.append(vocab[start_token_id])
        current_len -= 1

    input = torch.LongTensor([[start_token_id]]).to(device)

    with torch.no_grad():
        for i in range(line_num):
            for interpunction in ["，", "。\n"]:
                while current_len > 0:
                    output, _ = model(input)
                    prob = torch.softmax(output[0, 0], dim=-1)
                    # 按照概率进行随机选取
                    next_token = torch.multinomial(prob, num_samples=1)
                    # save
                    poem.append(vocab[next_token.item()])
                    input = next_token.unsqueeze(0)
                    current_len -= 1

                poem.append(interpunction)
                current_len = line_length

    return "".join(poem)

print(generate(model, vocab, word2id, start_token="一", line_num=4, line_length=7))
