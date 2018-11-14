from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from seq2seq import Seq2SeqSummarizer
from fit import fit_text
import numpy as np

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42)
    data_dir_path = './data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/news_summary.csv", encoding = 'cp437')
    
    df = df.dropna()
    df = df.drop(['date','headlines','read_more'],1)
    df = df.set_index('author')
    df = df.reset_index(drop=True)

    print('extract configuration from input texts ...')
    Y = df.text
    X = df.ctext

    config = fit_text(X, Y)

    summarizer = Seq2SeqSummarizer(config)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(summarizer.version) + '.png'


if __name__ == '__main__':
    main()