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

MODEL_PATH = "langtrans-00(RNN, LSTM).pt"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load dataset
dataset = datasets.load_dataset("Helsinki-NLP/open_subtitles", lang1="en", lang2="ko")
eng_texts = list(
    map(
        lambda x: x["translation"]["en"],
        tqdm(dataset["train"], desc="Loading English"),
    ),
)
kor_texts = list(
    map(
        lambda x: x["translation"]["ko"],
        tqdm(dataset["train"], desc="Loading Korean"),
    ),
)
# %%
# Tokenizer
tokenizer_eng = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
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
        kor_tokenized = tokenizer_eng(
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
            "eng_in": eng_input,
            "kor_in": kor_input,
            "kor_trg": kor_target,
        }


dataset = TranslationDataset(eng_texts, kor_texts, tokenizer_eng, tokenizer_kor, 128)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# %%
# model


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, c) = self.lstm(x)
        return h, c


# Decoder 정의
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.fc = nn.Linear(latent_dim, vocab_size)

    def forward(self, x, initial_states):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x, initial_states)
        output = self.fc(lstm_out)
        return output


# Seq2Seq 정의
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        encoder_states = self.encoder(encoder_input)
        output = self.decoder(decoder_input, encoder_states)
        return output


# %%
#  모델 초기화
embedding_dim = 256
latent_dim = 256
output_dim = len(tokenizer_kor.vocab)

encoder = Encoder(tokenizer_eng.vocab_size, embedding_dim, latent_dim)
decoder = Decoder(tokenizer_kor.vocab_size, embedding_dim, latent_dim)
model = Seq2Seq(encoder, decoder)

model.to(device)
with SummaryWriter() as writer:
    first_batch = next(iter(dataloader))
    seq_eng_input = first_batch["eng_in"].to(device)
    seq_kor_input = first_batch["kor_in"].to(device)
    writer.add_graph(
        model, [seq_eng_input[:10].to(device), seq_kor_input[:10].to(device)]
    )
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
num_epochs = 1
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

                del eng_in, kor_in, kor_trg, output
# %%
# 모델 저장
torch.save(model.state_dict(), MODEL_PATH)


# %%
def token_postprocessing(token_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = list(
        itertools.takewhile(
            lambda x: x != "[SEP]",
            filter(lambda x: x != "[PAD]", filter(lambda x: x != "[CLS]", tokens)),
        )
    )
    return tokenizer.convert_tokens_to_string(tokens)


# 평가 (예제, 학습 후 모델을 평가하는 방법)
model.eval()
with torch.no_grad():
    # 램덤하게 선택
    index = np.random.randint(0, len(seq_eng_input))
    print(f"index : {index} / {len(seq_eng_input)}")
    sample_eng = seq_eng_input[index].unsqueeze(0).to(device)
    sample_kor = seq_kor_input[index].unsqueeze(0).to(device)

    output = model(sample_eng, sample_kor)
    predicted_ids = torch.argmax(output, dim=-1)
    predicted_tokens = tokenizer_kor.convert_ids_to_tokens(predicted_ids[0])

    origin_eng = token_postprocessing(sample_eng[0], tokenizer_eng)
    origin_kor = token_postprocessing(sample_kor[0], tokenizer_kor)
    predict = token_postprocessing(predicted_ids[0], tokenizer_kor)

    print(f"(원본) 영어   : {origin_eng}")
    print(f"(원본) 한국어 : {origin_kor}")
    print(f"(예측) 한국어 : {predict}")


# %%
def translate(sentence: str) -> str:
    # Tokenize the input English sentence
    seq_eng = tokenizer_eng(
        [sentence],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=False,
    )
    seq_eng_input = seq_eng["input_ids"].to(device)

    # Get the initial states from the encoder
    encoder_states = model.encoder(seq_eng_input)

    # Initialize the decoder input with the start token
    start_token = tokenizer_kor.cls_token_id
    decoder_input = torch.tensor([[start_token]], device=device)

    generated_tokens = []

    # Iteratively generate tokens
    with torch.no_grad():
        for _ in range(128):  # Maximum sequence length
            output = model.decoder(decoder_input, encoder_states)
            next_token_logits = output[:, -1, :]  # Get the last token prediction
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_tokens.append(next_token.item())

            if next_token.item() == tokenizer_kor.sep_token_id:  # End token
                break

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

    # Postprocess the generated tokens
    translation = token_postprocessing(generated_tokens, tokenizer_kor)
    return translation


# Test the translate function
translated_text = translate("Hello, world!")
print(f"(원본) 영어   : Hello, world!")
print(f"(예측) 한국어 : {translated_text}")
