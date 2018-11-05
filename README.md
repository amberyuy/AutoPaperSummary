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
  * Summary.py: The code to get the extracted-sentences and key-words.
  * textrank.py: The code of textrank.
  * setup.py：the environment should be set
  * textrank.extract_sentences() is used to get summary of the file
  * textrank.extract_key_phrases() is used to get keywords of the file
* Setting Environment
  * install python3
  * install TensorFlow1.10,numpy,pandas,nltk,editdistance OR run the setup.py(Alternatively, if you have access to pip you may install the library directly from github:)
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
  * [Word2vec](https://en.wikipedia.org/wiki/Word2vec) algorithm [skipgram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) is used for the encoder input sequence.Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space
![word2vec](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization-/raw/master/skip-gram.jpg)
* Encoder-Decoder Model
  * Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.
  * Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.
![Encoder-Decoder](https://cdn-images-1.medium.com/max/1585/1*sO-SP58T4brE9EHazHSeGA.png)

* Recurrent Neural Networks(RNN) with LSTM
  * [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network):RNN performs the same task for every element of a sequence, with the output being depended on the previous computations,and they have a “memory” which captures information about what has been calculated so far.
![RNN model](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
  * [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory):An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.This [article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) can help you better understand LSTM
![LSTM model](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
* [Attention Model](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
  * it proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step. It is proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step.
![attention layer](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization-/raw/master/BiEnDeLstmAttention.jpg)

#### Dataset
* we use the reviews written about fine foods sold on Amazon. This dataset contains above 500,000 reviews, and is hosted on [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews/data)

#### Usage
* Files/Functions:
  * summarize_reviews.ipynb：the whole jupyter file to do the review summary
  
* Pre-req:
  * install python3
  * install TensorFlow1.10,
  * install numpy,nltk

* Current results
  * We train it by GPU and then add train outcome to Google Drive,because the docunments are too big, so you can downloads them and try to change file path and only run the last step of the code in summarize_reviews.ipynb. But I have to say the outcome is not very satisfactory, so we are changing the model's parameters and improving it. [train model](https://drive.google.com/open?id=1yA4jbxyPpHEvyH7rmdyqeXkAjsyEIeOr)
