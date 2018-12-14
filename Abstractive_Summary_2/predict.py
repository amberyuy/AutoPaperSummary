
import pandas as pd
from train import Seq_to_Seq
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    df_s = pd.read_csv(data_dir_path + '/sports.csv', encoding='utf-8')

    print('extract configuration from input texts ...')
    Y = df_s.summary
    X = df_s.articles

    config = np.load(Seq_to_Seq.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq_to_Seq(config)
    summarizer.load_weights(weight_file_path=Seq_to_Seq.get_weight_file_path(model_dir_path=model_dir_path))

    for i in np.random.permutation(np.arange(len(X)))[0:20]:
        x = X[i]
        original_summary = Y[i]
        summary = summarizer.summarize(x)
        print('=======================================')
        print('Original text: ', x)
        print('=======================================')
        print('Original news_summary: ', original_summary)
        print('=======================================')
        print('Generated news_summary: ', summary)
        print(" ")


if __name__ == '__main__':
    main()
