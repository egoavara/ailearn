# %%
# import packages
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import numpy as np
import transformers
import datasets

# %%
# load dataset
dataset = datasets.load_dataset("didi0di/Chatbot_data_for_Korean_v1.0")

eng_texts = []
kor_inputs = []
kor_targets = []

# %%
dset0 = load_dataset("Helsinki-NLP/open_subtitles", lang1="en", lang2="ko")
# %%
for i, data in enumerate(dset0["train"]):
    if i % 1000 == 0:
        print(f"Processing... ({i}/{len(dset0['train'])})")
    prompt = data["translation"]["en"]
    chosen = data["translation"]["ko"]
    eng_texts.append(prompt)
    kor_inputs.append("<sos> " + chosen)
    kor_targets.append(chosen + " <eos>")
# %%
# hyperparameters
num_words_eng = 10000
embedding_dim = 100
num_words_kor = 10000
latent_dim = 256
max_len_kor = 20
# %%
# model


class Encoder(nn.Module):
    def __init__(self, num_words_eng, embedding_dim, embedding_matrix, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_words_eng, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, num_words_kor, embedding_dim, latent_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_words_kor, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.dense = nn.Linear(latent_dim, num_words_kor)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, initial_states):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x, initial_states)
        output = self.dense(lstm_out)
        output = self.softmax(output)
        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        encoder_states = self.encoder(encoder_input)
        output = self.decoder(decoder_input, encoder_states)
        return output


embedding_matrix = np.random.rand(num_words_eng, embedding_dim)
encoder = Encoder(num_words_eng, embedding_dim, embedding_matrix, latent_dim)
decoder = Decoder(num_words_kor, embedding_dim, latent_dim)

model = Seq2Seq(encoder, decoder)

# %%
# loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 is the padding index
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
# Summary (not as simple as in Keras, using print statements)
print(model)
