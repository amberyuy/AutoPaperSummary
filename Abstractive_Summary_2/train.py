from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from collections import Counter

Load_Weight = False

Maxinseq = 2000
Maxtarseq = 200
Maxinv = 5000
Maxtarv = 2000

Hid_unit = 300
Batchsize = 64
Verbose = 1
Epoch = 10


def Prepro(X, Y, seqin_max=None, seqtar_max=None):
    if seqin_max is None:
        seqin_max = Maxinseq
    if seqtar_max is None:
        seqtar_max = Maxtarseq
    incount = Counter()
    target_counter = Counter()
    max_inseq = 0
    maxtarseq = 0

    for line in X:
        text = [word.lower() for word in line.split(' ')]
        seq_length = len(text)
        if seq_length > seqin_max:
            text = text[0:seqin_max]
            seq_length = len(text)
        for word in text:
            incount[word] += 1
        max_inseq = max(max_inseq, seq_length)

    for line in Y:
        line2 = 'START ' + line.lower() + ' END'
        text = [word for word in line2.split(' ')]
        seq_length = len(text)
        if seq_length > seqtar_max:
            text = text[0:seqtar_max]
            seq_length = len(text)
        for word in text:
            target_counter[word] += 1
            maxtarseq = max(maxtarseq, seq_length)

    in_word2index = dict()
    for idx, word in enumerate(incount.most_common(Maxinv)):
        in_word2index[word[0]] = idx + 2
    in_word2index['PAD'] = 0
    in_word2index['UNK'] = 1
    in_indextoword = dict([(idx, word) for word, idx in in_word2index.items()])

    tar_word2index = dict()
    for idx, word in enumerate(target_counter.most_common(Maxtarv)):
        tar_word2index[word[0]] = idx + 1
    tar_word2index['UNK'] = 0

    tar_index2word = dict([(idx, word) for word, idx in tar_word2index.items()])

    intoken_num = len(in_word2index)
    tartok_num = len(tar_word2index)

    config = dict()
    config['in_word2index'] = in_word2index
    config['in_indextoword'] = in_indextoword
    config['tar_word2index'] = tar_word2index
    config['tar_index2word'] = tar_index2word
    config['intoken_num'] = intoken_num
    config['tartok_num'] = tartok_num
    config['max_inseq'] = max_inseq
    config['maxtarseq'] = maxtarseq

    return config


class Seq_to_Seq():

    def __init__(self, config):
        self.model_name = 'seq2seq'
        self.intoken_num = config['intoken_num']
        self.max_inseq = config['max_inseq']
        self.tartok_num = config['tartok_num']
        self.maxtarseq = config['maxtarseq']
        self.in_word2index = config['in_word2index']
        self.in_indextoword = config['in_indextoword']
        self.tar_word2index = config['tar_word2index']
        self.tar_index2word = config['tar_index2word']
        self.config = config
        self.version = 0

        if 'version' in config:
            self.version = config['version']

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.intoken_num, output_dim=Hid_unit,
                                      input_length=self.max_inseq, name='encoder_embedding')
        encoder_lstm = LSTM(units=Hid_unit,kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), dropout = 0.5, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.tartok_num), name='decoder_inputs')
        decoder_lstm = LSTM(units=Hid_unit,kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(units=self.tartok_num, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(Hid_unit,)), Input(shape=(Hid_unit,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def tranintext(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.in_word2index:
                    wid = self.in_word2index[word]
                x.append(wid)
                if len(x) >= self.max_inseq:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_inseq)

        print(temp.shape)
        return temp

    def trans_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.maxtarseq:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_inseq)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.maxtarseq, self.tartok_num))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.maxtarseq, self.tartok_num))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0
                        if w in self.tar_word2index:
                            w2idx = self.tar_word2index[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

        @staticmethod
        def get_weight_file_path(model_dir_path):
            return model_dir_path + '/' + self.model_name + '-weights.h5'

        @staticmethod
        def get_config_file_path(model_dir_path):
            return model_dir_path + '/' + self.model_name + '-config.npy'

        @staticmethod
        def get_architecture_file_path(model_dir_path):
            return model_dir_path + '/' + self.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = Epoch
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = Batchsize

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq_to_Seq.get_config_file_path(model_dir_path)
        weight_file_path = Seq_to_Seq.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq_to_Seq.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.trans_target_encoding(Ytrain)
        Ytest = self.trans_target_encoding(Ytest)

        Xtrain = self.tranintext(Xtrain)
        Xtest = self.tranintext(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=Verbose, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.in_word2index:
                idx = self.in_word2index[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_inseq)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.tartok_num))
        target_seq[0, 0, self.tar_word2index['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.tar_index2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.maxtarseq:
                terminated = True

            target_seq = np.zeros((1, 1, self.tartok_num))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    df_s = pd.read_csv(data_dir_path + '/sports.csv', encoding='utf-8')

    # print('extract configuration from input texts ...')
    Ys = df_s.summary
    Xs = df_s.articles

    df_e = pd.read_csv(data_dir_path + '/entertainment.csv', encoding='utf-8')

    #print('extract configuration from input texts ...')
    Ye = df_e.summary
    Xe = df_e.articles

    df_b = pd.read_csv(data_dir_path + '/business.csv', encoding='utf-8')

    #print('extract configuration from input texts ...')
    Yb = df_b.summary
    Xb = df_b.articles

    df_t = pd.read_csv(data_dir_path + '/tech.csv', encoding='utf-8')

    #print('extract configuration from input texts ...')
    Yt = df_t.summary
    Xt = df_t.articles

    df_p = pd.read_csv(data_dir_path + '/politics.csv', encoding='utf-8')

    # print('extract configuration from input texts ...')
    Yp = df_p.summary
    Xp = df_p.articles

    framey = [Ys, Ye, Yb, Yt, Yp]
    Y = pd.concat(framey)
    framex = [Xs, Xe, Xb, Xt, Xp]
    X = pd.concat(framex)
    config = Prepro(X, Y)

    summarizer = Seq_to_Seq(config)

    if Load_Weight:
        summarizer.load_weights(weight_file_path=Seq_to_Seq.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')


if __name__ == '__main__':
    main()
