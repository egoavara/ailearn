# %%
# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import os
import polars as pl
import numpy as np
import transformers
import datasets
import itertools
from tqdm import tqdm

MODEL_PATH = "langtrans-00(transformer).pt"

# Check if CUDA is available
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# %%
# load dataset
dataset = datasets.load_dataset(
    "Helsinki-NLP/open_subtitles", lang1="en", lang2="ko")
eng_texts = list(
    map(
        lambda x: x["translation"]["en"],
        # tqdm(dataset["train"], desc="Loading English"),
        itertools.islice(tqdm(dataset["train"], desc="Loading English"), 10000),
    ),
)
kor_texts = list(
    map(
        lambda x: x["translation"]["ko"],
        # tqdm(dataset["train"], desc="Loading Korean"),
        itertools.islice(tqdm(dataset["train"], desc="Loading Korean"), 10000),
    ),
)
# %%
# Tokenizer
tokenizer_eng = transformers.BertTokenizerFast.from_pretrained(
    "bert-base-uncased")
tokenizer_kor = transformers.BertTokenizerFast.from_pretrained(
    "bert-base-multilingual-cased"
)


# 데이터셋 정의
class TranslationDataset(Dataset):
    def __init__(
        self, eng_texts, kor_texts, tokenizer_eng, tokenizer_kor, token_max=128
    ):
        self.eng_texts = eng_texts
        self.kor_texts = kor_texts
        self.tokenizer_eng = tokenizer_eng
        self.tokenizer_kor = tokenizer_kor
        self.token_max = token_max

    def __len__(self):
        return min(len(self.eng_texts), len(self.kor_texts))

    def __getitem__(self, idx):

        eng_tokenized = tokenizer_eng(
            self.eng_texts[idx],
            padding=True,
            truncation=True,
            max_length=self.token_max,
            return_tensors="pt",
            add_special_tokens=False,
        )
        kor_tokenized = tokenizer_kor(
            self.kor_texts[idx],
            padding=True,
            truncation=True,
            max_length=self.token_max,
            return_tensors="pt",
            add_special_tokens=True,
        )

        eng_input = eng_tokenized["input_ids"].squeeze(0)
        kor_input = kor_tokenized["input_ids"].squeeze(0)[:-1]
        kor_target = kor_tokenized["input_ids"].squeeze(0)[1:]

        # eng_input = F.pad(eng_input, (0, 128 - eng_input.size(0)), value=self.tokenizer_eng.pad_token_id)[:self.max_length]
        eng_input = F.pad(
            eng_input,
            (self.token_max - eng_input.size(0), 0),
            value=self.tokenizer_eng.pad_token_id,
        )
        kor_input = F.pad(
            kor_input,
            (0, self.token_max - kor_input.size(0)),
            value=self.tokenizer_kor.pad_token_id,
        )
        kor_target = F.pad(
            kor_target,
            (0, self.token_max - kor_target.size(0)),
            value=self.tokenizer_kor.pad_token_id,
        )
        return {
            "eng_in": eng_input.to(torch.int64),
            "kor_in": kor_input.to(torch.int64),
            "kor_trg": kor_target.to(torch.int64),
        }


dataset = TranslationDataset(
    eng_texts, kor_texts, tokenizer_eng, tokenizer_kor, 128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# %%
# model


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Parameter(torch.zeros(
            1, 128, embedding_dim))  # Assume max length of 500
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Parameter(torch.zeros(
            1, 128, embedding_dim))  # Assume max length of 500
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, memory):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]
        x = self.transformer_decoder(x, memory)
        output = self.fc(x)
        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        memory = self.encoder(encoder_input)
        output = self.decoder(decoder_input, memory)
        return output


# %%
#  모델 초기화
embedding_dim = 256
num_heads = 8
num_layers = 3
ff_dim = 512
output_dim = len(tokenizer_kor.vocab)

encoder = TransformerEncoder(
    tokenizer_eng.vocab_size, embedding_dim, num_heads, num_layers, ff_dim)
decoder = TransformerDecoder(
    tokenizer_kor.vocab_size, embedding_dim, num_heads, num_layers, ff_dim)
model = Seq2Seq(encoder, decoder)

model.to(device)
with SummaryWriter() as writer:
    first_batch = next(iter(dataloader))
    seq_eng_input = first_batch["eng_in"].to(device)
    seq_kor_input = first_batch["kor_in"].to(device)
    writer.add_graph(model, [seq_eng_input[:10], seq_kor_input[:10]])
    del seq_eng_input, seq_kor_input
    torch.cuda.empty_cache()
print(model)
if os.path.exists(MODEL_PATH):
    print("Loading model")
    model.load_state_dict(torch.load(MODEL_PATH))

# %%
# 학습 루프
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer_kor.pad_token_id
)  # assuming 0 is the padding index
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
model.train()
with SummaryWriter() as writer:
    for epoch in range(num_epochs):
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                eng_in = batch["eng_in"].to(device)
                kor_in = batch["kor_in"].to(device)
                kor_trg = batch["kor_trg"].to(device)

                optimizer.zero_grad()

                output = model(eng_in, kor_in)
                output = output.view(-1, output_dim)
                kor_trg = kor_trg.view(-1)

                loss = criterion(output, kor_trg)

                writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                optimizer.step()
# %%
# 모델 저장
torch.save(model.state_dict(), MODEL_PATH)


# %%
def token_postprocessing(token_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = list(
        itertools.takewhile(
            lambda x: x != "[SEP]",
            filter(lambda x: x != "[PAD]", filter(
                lambda x: x != "[CLS]", tokens)),
        )
    )
    return tokenizer.convert_tokens_to_string(tokens)


# 평가 (예제, 학습 후 모델을 평가하는 방법)
model.eval()
with torch.no_grad():
    # 램덤하게 선택
    index = np.random.randint(0, len(dataset))
    print(f"index : {index} / {len(dataset)}")
    sample_eng = dataset[index]["eng_in"].unsqueeze(0).to(device)
    sample_kor = dataset[index]["kor_in"].unsqueeze(0).to(device)
    output = model(sample_eng, sample_kor)
    predicted_ids = torch.argmax(output, dim=-1)
    predicted_tokens = tokenizer_kor.convert_ids_to_tokens(predicted_ids[0])
    print(predicted_ids[0])
    print(predicted_tokens)
    origin_eng = token_postprocessing(sample_eng[0], tokenizer_eng)
    origin_kor = token_postprocessing(sample_kor[0], tokenizer_kor)
    predict = token_postprocessing(predicted_ids[0], tokenizer_kor)

    print(f"(원본) 영어   : {origin_eng}")
    print(f"(원본) 한국어 : {origin_kor}")
    print(f"(예측) 한국어 : {predict}")


# %%
def translate(sentence: str) -> str:
    model.eval()
    # Tokenize the input English sentence
    seq_eng = tokenizer_eng(
        sentence,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    seq_eng_input = seq_eng["input_ids"].squeeze(0)
    seq_eng_input = F.pad(
        seq_eng_input, (128 - seq_eng_input.size(0), 0),
        value=tokenizer_eng.pad_token_id
    )
    seq_eng_input = seq_eng_input.unsqueeze(0)
    seq_eng_input = seq_eng_input.to(device)
    print(f"Encoder input shape: {seq_eng_input.shape}")
    print(f"Encoder input shape: {seq_eng_input}")

    # Encode the input sentence
    encoder_states = model.encoder(seq_eng_input)
    print(f"Encoder output shape: {encoder_states.shape}")

    # Initialize the decoder input with the start token and pad to length 128
    decoder_input = torch.full(
        (1, 128), tokenizer_kor.pad_token_id, device=device)
    decoder_input[0, 0] = tokenizer_kor.cls_token_id
    print(f"Initial decoder input shape: {decoder_input}")
    print(f"Initial decoder input shape: {decoder_input.shape}")

    generated_tokens = []

    # Iteratively generate tokens
    with torch.no_grad():
        for _ in range(128):  # Maximum sequence length
            output = model.decoder(decoder_input, encoder_states)
            print(f"Decoder output shape: {output.shape}")
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            print(f"Decoder output: {next_token}")

            print(f"Next token: {next_token.item()}")


            if next_token.item() == tokenizer_kor.sep_token_id:
                break

            generated_tokens.append(next_token.item())
            # Update the decoder input
            decoder_input = torch.roll(decoder_input, -1, dims=1)
            decoder_input[0, -1] = next_token
    print(f"Generated tokens: {generated_tokens}")
    # Postprocess the generated tokens
    translation = token_postprocessing(generated_tokens, tokenizer_kor)
    return translation


# Example usage
translated_text = translate("it' s time to ring the bell")
print(f"Translated Korean: {translated_text}")
