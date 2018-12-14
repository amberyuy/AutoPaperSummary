#!/usr/bin/python
from __future__ import print_function

import pandas as pd
from train import Seq_to_Seq
import numpy as np

np.random.seed(42)
data_dir_path = './data'
model_dir_path = './models' 

with open('./predict.txt', 'r') as myfile:
  X = myfile.read()

process = np.load(Seq_to_Seq.get_config_file_path(model_dir_path=model_dir_path)).item()

summarizer = Seq_to_Seq(process)
summarizer.load_weights(weight_file_path=Seq_to_Seq.get_weight_file_path(model_dir_path=model_dir_path))
x = X
summary = summarizer.summarize(X)
with open('result.txt','w') as wf:
  file.write(summary)


print("=======================================================")
print('Article: ', x)
print("=======================================================")
print('Generated Summary: ', summary)


