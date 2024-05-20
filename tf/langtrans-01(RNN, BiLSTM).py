# %%

import numpy as np
import polars as pl

import tensorflow as tf

import keras.api as keras
from keras.api.models import Model
from keras.api.layers import Dense, Embedding, Input, LSTM, Bidirectional, Concatenate, BatchNormalization
from keras.api.utils import to_categorical
from keras.api.optimizers import Adam
from keras.api.utils import plot_model
from keras.api.layers import Attention
from keras.api.callbacks import ModelCheckpoint
from keras.api.regularizers import l2

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

# %%
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
LATENT_DIM = 512
# %%
file_path = keras.utils.get_file(
    "kor.txt", "https://github.com/ironmanciti/NLP_lecture/raw/master/data/kor.txt")

eng_texts = []
kor_inputs = []
kor_targets = []

for line in open(file_path, 'r', encoding='utf-8'):

    if '\t' not in line:     # \t 가 없는 line 은 skip
        continue

    # input 과 target translation 구분
    english, korean, attribution = line.split('\t')

    # target input 과 output 을 teacher forcing 입력 구성
    input = '<sos> ' + korean
    target = korean + ' <eos>'

    eng_texts.append(english)     # 영어 문장

    kor_inputs.append(input)
    kor_targets.append(target)


# %% 영어 Tokenizer
tokenizer_eng = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer_eng.fit_on_texts(eng_texts)
eng_sequences = tokenizer_eng.texts_to_sequences(eng_texts)

word2idx_eng = tokenizer_eng.word_index
print(f'unique input token 수 : {len(word2idx_eng)}')
num_words_eng = min(MAX_VOCAB_SIZE, len(word2idx_eng) + 1)
print("Input Text 의 단어 수 :", num_words_eng)
max_len_eng = max(len(s) for s in eng_sequences)
print("Input Text 의 최대 길이 :", max_len_eng)

# %% 한국어 Tokenizer
tokenizer_kor = Tokenizer(num_words=MAX_VOCAB_SIZE,  filters="")
tokenizer_kor.fit_on_texts(kor_inputs + kor_targets)

kor_input_sequences = tokenizer_kor.texts_to_sequences(kor_inputs)
kor_target_sequences = tokenizer_kor.texts_to_sequences(kor_targets)

word2idx_kor = tokenizer_kor.word_index
print(f'unique output tokens : {len(word2idx_kor)}')
num_words_kor = len(word2idx_kor) + 1
print("Target 언어의 단어 수 :", num_words_kor)
max_len_kor = max(len(s) for s in kor_target_sequences)
print("Target 언어의 최대 길이 :", max_len_kor)
# %%
print(tokenizer_kor.sequences_to_texts(kor_input_sequences[1500:1501]))
print(tokenizer_kor.sequences_to_texts(kor_target_sequences[1500:1501]))

einputs = pad_sequences(
    eng_sequences, maxlen=max_len_eng, padding="pre")
print("encoder input shape :", einputs.shape)
print("encoder_inputs[0] : ", einputs[1500])

dinputs = pad_sequences(
    kor_input_sequences, maxlen=max_len_kor, padding="post")
print("\ndecoder input shape :", dinputs.shape)
print("decoder_inputs[0] : ", dinputs[1500])

dtargets = pad_sequences(
    kor_target_sequences, maxlen=max_len_kor, padding="post")
print("\nencoder target shape :", dtargets.shape)
print("encoder_targets[0] : ", dtargets[1500])

# %%


def make_embedding(num_words, embedding_dim, tokenizer, max_vocab_size):

    embeddings_dict = {}
    output = '../glove.6B.100d.txt'

    with open(output, encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()            # 각 줄을 읽어와서 word_vector에 저장
            word = values[0]                 # 첫번째 값은 word
            # 두번째 element 부터 마지막까지 100 개는 해당 단어의 임베딩 벡터의  값
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs

    embedding_matrix = np.zeros((num_words, embedding_dim))    # zero 로 초기화

    print("word 갯수 =", num_words)
    print(embedding_matrix.shape)

    for word, i in tokenizer.word_index.items():
        if i < max_vocab_size:
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:         # 해당 word 가 없으면 all zero 로 남겨둠
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


num_words = min(MAX_VOCAB_SIZE, len(word2idx_kor) + 1)
num_words
# %%

embedding_matrix = make_embedding(num_words_eng, EMBEDDING_DIM,
                                  tokenizer_eng, MAX_VOCAB_SIZE)
# %%

# Encoder
encoder_inputs_ = Input(shape=(max_len_eng, ), name='Encoder_Input')

# pre-trained embedding layer 사용
embedding_layer = Embedding(num_words_eng, EMBEDDING_DIM,
                            weights=[embedding_matrix], trainable=True)
x = embedding_layer(encoder_inputs_)
encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
    LSTM(LATENT_DIM, return_state=True, kernel_regularizer=l2(0.01)))(x)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
# encoder 는 hidden state and cell state 만 decoder 로 전달 --> thought vector
encoder_states = [state_h, state_c]

encoder_model = Model(encoder_inputs_, encoder_states)
encoder_model.summary()
decoder_inputs_ = Input(shape=(max_len_kor,), name="Decoder_Input")

# decoder word embedding 은 pre-trained vector 를 사용 않음
decoder_embedding = Embedding(num_words_kor, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_)

# decoder for teacher-forcing
decoder_lstm = LSTM(LATENT_DIM * 2, return_sequences=True, return_state=True, kernel_regularizer=l2(0.01))
# initial state = encoder [h, c]
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x,
                                     initial_state=encoder_states)
# final layer
decoder_dense = Dense(num_words_kor, activation='softmax',
                      name='Decoder_Output')
decoder_outputs = decoder_dense(decoder_outputs)

# Teacher-forcing 모델 생성
model_teacher_forcing = Model(
    [encoder_inputs_, decoder_inputs_],  decoder_outputs)

# model compile and train
model_teacher_forcing.compile(loss='sparse_categorical_crossentropy',
                              optimizer=Adam(0.001), metrics=['accuracy'])

model_teacher_forcing.summary()
plot_model(model_teacher_forcing, show_shapes=True)

# %%
checkpoint = ModelCheckpoint(
    "langtrans-01(RNN, BiLSTM).keras", save_best_only=True, monitor='val_loss', mode='min')
history = model_teacher_forcing.fit([einputs, dinputs],
                                    dtargets, batch_size=BATCH_SIZE,
                                    epochs=30, validation_split=0.1, callbacks=[checkpoint])
# %%
df = pl.DataFrame(history.history)
# %%
df.plot.line(y=['loss', 'val_loss'], title='Loss', height=600, width=1000,)
# %%
df.plot.line(y=['accuracy', 'val_accuracy'], ylim=(0.9, 1.),
             height=600, width=1000, title='Accuracy')

# %%

# Decoder for inference
decoder_state_input_h = Input(shape=(1024,), name='Decoder_hidden_h')
decoder_state_input_c = Input(shape=(1024,), name='Decoder_hidden_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,), name='Decoder_input')
x = decoder_embedding(decoder_inputs_single)

# output, hidden states 를 저장
decoder_outputs, h, c = decoder_lstm(x, initial_state=decoder_states_inputs)

decoder_states = [h, c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

decoder_model.summary()
plot_model(decoder_model, show_shapes=True)

# %%


def decode_sequence(input_seq):
   # encoder model을 이용하여 input을 state vector로 encoding
    states_value = encoder_model.predict(input_seq)

   # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

   # target sequence 의 첫번째 character 를 start character (<sos>) 로 설정
    target_seq[0, 0] = word2idx_kor['<sos>']

   # <eos> token이 decode 에서 생성되면 loop 에서 break
    eos = word2idx_kor['<eos>']

   # 번역문 생성
    output_sentence = []
    for _ in range(max_len_kor):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

       # argmax 로 가장 확률 높은 단어 선택 --> greedy selection
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:  # <EOS> token
            break

        if idx > 0:  # idx 0 은 zero padding 된 sequence 이므로 ''
            word = tokenizer_kor.index_word[idx]
            output_sentence.append(word)

        # 생성된 word 를 decoder 의 다음 input 으로 사용
        target_seq[0, 0] = idx

       # Update states
        states_value = [h, c]

    return ' '.join(output_sentence)


# %%
for _ in range(5):
    i = np.random.choice(len(eng_texts))
    input_seq = einputs[i:i+1]

    translation = decode_sequence(input_seq)
    print('-')
    print('Input:', eng_texts[i])
    print('Translation:', translation)
# %%


def Eng_Kor_translation(txt):
    input_sequence = tokenizer_eng.texts_to_sequences([txt])
    encoder_input = pad_sequences(input_sequence, maxlen=max_len_eng)

    return decode_sequence(encoder_input)


test_text = [
    "Your lips are red.",
    "French is interesting.",
    "I like you.",
    "Let's go to home."
]
for text in test_text:
    translation = Eng_Kor_translation(text)
    print('----')
    print('Input:', text)
    print('Translation:', translation)
