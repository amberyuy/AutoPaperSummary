# Automatic Paper Summary
![paper summary](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/image_1.png)

## Product Statement
Text summarization is the problem of creating a short, accurate, and fluent summary of a longer text document.
Here is an [Introduction to text summary](https://machinelearningmastery.com/gentle-introduction-text-summarization/)

## Basic approach
Text summarization can broadly be divided into two categories — **Extractive Summarization** and **Abstractive Summarization**
### Extractive Summarization
* Select relevant phrases of the input document and concatenate them to form a summary (like "copy-and-paste").
  * Pros: They are quite robust since they use existing natural-language phrases that are taken straight from the input.
  * Cons: they lack in flexibility since they cannot use novel words or connectors. They also cannot paraphrase like people sometimes do

#### <font color=#00ffff> TextRank </font>
**[TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)** is based on [PageRank](https://en.wikipedia.org/wiki/PageRank) algorithm that is used on Google Search Engine. Its base concept is "The linked page is good, much more if it from many linked page". In TextRank, article is divided into basic text units, i.e., words or phrases. As treated as webpage in PageRank, text unit maps to vertex in graph, and edge between vertexes refers to the link between text units.
The Classic PageRank algorithm workflow is as below:
![PageRank](https://github.com/icoxfog417/awesome-text-summarization/raw/master/images/page_rank.png)

#### Usage
* Files/Functions:
  * [Summary.py](https://github.com/jyueling/601Project/blob/master/TextRank/summary.py): The code to get the extracted-sentences and key-words.
  * [textrank.py](https://github.com/jyueling/601Project/blob/master/TextRank/textrank.py): The code of textrank.
  * [setup.py](https://github.com/jyueling/601Project/blob/master/TextRank/setup.py)：the environment should be set
* Setting Environment
  * install python3
  * install TensorFlow1.10,numpy,pandas,nltk,editdistance OR run the setup.py(Alternatively, if you have access to pip you may install the library directly from github)
  ```
  pip install git+git://github.com/davidadamojr/TextRank.git
  ```
* How to run
  * Put the text in the article folder
  * Run in Terminal
  ```
  textrank extract_summary <filename>
  ```




### Abstractive Summarization
* Generate a summary that keeps original intent. It's just like humans do
  * Pros: They can use words that were not in the original input. It enables to make more fluent and natural summaries.
  * Cons: But it is also a much harder problem as you now require the model to generate coherent phrases and connectors.

#### Sequence-to-Sequence with Attention Model for Text Summarization
To build our model,we will use a two-layered bidirectional RNN with LSTMs on the input data for the encoder layer and two layers, each with an LSTM using attention on the target data for the decoder.This model is based on Xin Pan’s and Peter Liu’s model[(Github)](https://github.com/tensorflow/models/tree/master/research/textsum).Here is a good [article](https://towardsdatascience.com/text-summarization-with-amazon-reviews-41801c2210b) for this model and explains parts of the codes in detail.

#### Technology Selection
* Word Embedding
  * Distributional vectors or word embeddings essentially follow the distributional hypothesis, according to which words with similar meanings tend to occur in similar context. Thus, these vectors try to capture the characteristics of the neighbors of a word.Word embeddings are often used as the first data processing layer in a deep learning model.Word embeddings were revolutionized by Mikolov who proposed the [CBOW](https://iksinc.online/tag/continuous-bag-of-words-cbow/) and [skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) models
    * [Word2vec](https://en.wikipedia.org/wiki/Word2vec) is used for the encoder input sequence.Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space
![word2vec](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization-/raw/master/skip-gram.jpg)
    * [GloVe(Global Vectors for Word Representation）](https://nlp.stanford.edu/pubs/glove.pdf) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.The GloVe model is trained on the non-zero entries of a global word-word co-occurrence matrix, which tabulates how frequently words co-occur with one another in a given corpus. Populating this matrix requires a single pass through the entire corpus to collect the statistics.

* LSTM Model
  * [Recurrent Neural Network(RNN)](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912):Basic RNNs are a network of neuron-like nodes organized into successive "layers", each node in a given layer is connected with a directed (one-way) connection to every other node in the next successive layer.Each node (neuron) has a time-varying real-valued activation. Each connection has a modifiable real-valued weight. Nodes are either input nodes (receiving data from outside the network), output nodes (yielding results), or hidden nodes (that modify the data en route from input to output).**However, standard RNN has the gradient vanishing or exploding problems. In order to overcome the issues, Long Short-term Memory network (LSTM) was developed and achieved superior per- formance (Hochreiter and Schmidhuber, 1997).**
  * [Long Short-term Memory(LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory):An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of **a cell, an input gate, an output gate and a forget gate.** The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.This [article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) can help you better understand LSTM.**The standard LSTM cannot detect which is the important part for aspect-level sentiment classification.In order to address this issue, we propose to use an attention mechanism that can capture the key part of sentence in response to a given aspect.**
  ![LSTM](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization-/raw/master/BiEnDeLstm_preview.jpeg)

* [Attention Model](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
  * it proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step. It is proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step.**The attention mechanism can concentrate on different parts of a sentence when different aspects are taken as input**
![attention layer](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization-/raw/master/BiEnDeLstmAttention.jpg)

* [Adam Optimization](https://arxiv.org/abs/1412.6980)
  * The algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.The initial value of the moving averages and beta1 and beta2 values close to 1.0 (recommended) result in a bias of moment estimates towards zero. This bias is overcome by first calculating the biased estimates before then calculating bias-corrected estimates.


### Abstractive Summary_1
#### Dataset
* we use the reviews written about fine foods sold on Amazon. This dataset contains above 500,000 reviews, and is hosted on [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews/data),you can download by yourself.

#### Usage
* Files/Functions:
  * [summarize_reviews.ipynb](https://github.com/jyueling/601Project/blob/master/Abstractive%20Summarization_1/summarize_reviews.ipynb)：the whole jupyter file to do the review summary
  
* Setting Environment:
  * install python3
  * install TensorFlow1.10
  * install numpy, nltk

* Current results
  * We train it by GPU and then add train outcome to Google Drive,because the docunments are too big, so you can download [train model](https://drive.google.com/open?id=1yA4jbxyPpHEvyH7rmdyqeXkAjsyEIeOr) and try to change file path and only run the last step of the code in summarize_reviews.ipynb. But I have to say the outcome is not very satisfactory, so we are changing the model's parameters and improving it. 
  
### Abstractive_Summary_2
#### Dataset
* We use the news summary from Kaggle, which contains the summarized article and the complete article text which can be used for the summarization task. You can download [here](https://www.kaggle.com/sunnysai12345/news-summary).
* Because of the accuracy is not good using the news-summary dataset,we find another dataset on Kaggle. [BBC News Summary](https://www.kaggle.com/pariza/bbc-news-summary),which contains more than 4000 articles and summaries. After downloading from website,you need to convert them to csv format first using numpy/pandas.

#### Usage
* Files/Functions:
  * [seq2seq_train.py](https://github.com/jyueling/601Project/blob/master/Abstractive_Summary_2/seq2seq_train.py):After setting the environment,you can run this file to train the model.You can change the dataset and epoches in this file.
  * [seq2seq_predict.py](https://github.com/jyueling/601Project/blob/master/Abstractive_Summary_2/seq2seq_predict.py):You can run this file after training or download our trained model,it will give you several examples of the training results which includes the generated summary and the original summary.
  * [main1sent.py](https://github.com/jyueling/601Project/blob/master/Abstractive_Summary_2/main1sent.py):You can open your own txt file to get your summary by running this file.
  * [seq2seq.py](https://github.com/jyueling/601Project/blob/master/Abstractive_Summary_2/seq2seq.py):The model we are using,you can add layers and change the parameters in this file,
  * [fit.py](https://github.com/jyueling/601Project/blob/master/Abstractive_Summary_2/fit.py):This file is doing the word embedding,you can change the sequence and word length before training.
  
* Setting Environment:
  * install python3
  * install TensorFlow1.10
  * install numpy
  * install pandas
  

### Problems we have met
* Datasets Selection

    In the first month, we chose the [Amazon Food Review Datasets](https://www.kaggle.com/snap/amazon-fine-food-reviews/data), it contains many reviews from the customer and summary words of reviews. But the reviews are too short for us, and their original summaries are only some words, so the model can't be used for long articles.  
    
    In the second month, we tried to use another method to train our model. We find another dataset called [news_summary](https://www.kaggle.com/sunnysai12345/news-summary), We trained it many times using different ways and parameters, but the results have no connection with the article ,and all the generated summaries seem related to "India" or "government". So after taliking with Prof.Osama and TAs, we found the quality of the dataset is not very good, it has many repeated contents. 
    
    Finally, we decided to use the [bbc news summary](https://www.kaggle.com/pariza/bbc-news-summary) which has been used for many people.Because the dataset is '.txt' format, we can't use directly. So we use python to convert them to csv format.The dataset has about 4500 datas.  
    From this problem, we learned that dataset selection is one of the most importtant thing of deep learning.
    
* Overfitting
 
## Reference
  Recent Trends in Deep Learning Based Natural Language Processing:https://arxiv.org/pdf/1708.02709.pdf

  Extractive Summarization: https://github.com/davidadamojr/TextRank 
  
  Abstractive Summarization1: https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews
  
  
