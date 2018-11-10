from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np

np.random.seed(42)
data_dir_path = './demo/data' # refers to the demo/data folder
model_dir_path = './demo/models' # refers to the demo/models folder

with open('predict.txt', 'r') as myfile:
  X = myfile.read()



config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()

summarizer = Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

x = X

summary = summarizer.summarize(X)
print("\n=======================================================")
print('Article: ', x)
print("Output------------------------------------------------")
print('Generated Headline: ', summary)



file = open('result.text','w')
file.write(headline)
