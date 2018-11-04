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
**[TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)** is based on PageRank algorithm that is used on Google Search Engine. Its base concept is "The linked page is good, much more if it from many linked page". In TextRank, article is divided into basic text units, i.e., words or phrases. As treated as webpage in PageRank, text unit maps to vertex in graph, and edge between vertexes refers to the link between text units.
The Classic TextRank algorithm workflow is as below:
![TextRank](/Users/mac/Desktop/1.jpg)

#### Usage
* Files:
  * The code of TextRank is Summary.py，you can run this code this code.
  * textrank.extract_sentences() is used to get summary of the file
  * textrank.extract_key_phrases() is used to get keywords of the file
* Pre-req
  * install python3
  * install TensorFlow1.10,numpy,pandas,nltk


### Abstractive Summarization
* Generate a summary that keeps original intent. It's just like humans do
  * Pros: They can use words that were not in the original input. It enables to make more fluent and natural summaries.
  * Cons: But it is also a much harder problem as you now require the model to generate coherent phrases and connectors.

#### Sequence-to-Sequence with Attention Model for Text Summarization
To build our model,we will use a two-layered bidirectional RNN with LSTMs on the input data for the encoder layer and two layers, each with an LSTM using bahdanau attention on the target data for the decoder.This model is based on Xin Pan’s and Peter Liu’s model[(Github)](https://github.com/tensorflow/models/tree/master/research/textsum).Here is a good [paper](https://towardsdatascience.com/text-summarization-with-amazon-reviews-41801c2210b) for this model and explains parts of the codes in detail.

#### Technology Selection
* Encoder-Decoder Model
  * Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.
  * Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.

* Recurrent Neural Networks(RNN) with LSTM
  * RNN
  * LSTM
* Attention Model

![Encoder-Decoder](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Encoder-Decoder-Recurrent-Neural-Network-Model.png)



#### Attention Model
* **Attention** is proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step. It is proposed as a solution to the limitation of the Encoder-Decoder model encoding the input sequence to one fixed length vector from which to decode each output time step. 

![Attention Model](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Example-of-Attention.png)



## Abstractive (Sprint 3)
  Generate a summary that keeps original intent. It's just like humans do.Based on the paper,“ A Review on Automatic Text Summarization Approaches ” https://thescipub.com/PDF/jcssp.2016.178.190.pdf, we choose to use RNN and CNN as our model.The code is in …,for details of the codes,you can read this paper.https://towardsdatascience.com/text-summarization-with-amazon-reviews-41801c2210b

  Dataset we use is Amazon Reviews Data:
  
  Review.csv(https://www.kaggle.com/snap/amazon-fine-food-reviews/data)

