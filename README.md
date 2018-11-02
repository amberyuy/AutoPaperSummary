# Project:Automatic Paper Summary
Text summarization is the problem of creating a short, accurate, and fluent summary of a longer text document.
For more introductions about text summarization, https://machinelearningmastery.com/gentle-introduction-text-summarization/

There are mainly two ways to make the summary. Extractive and Abstractive.
## Extractive (Sprint 2)
Select relevant phrases of the input document and concatenate them to form a summary (like "copy-and-paste").
TextRank is the typical graph based method. For information about the TextRank, https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
The code of TextRank is Summary.py，you can run this code this code.

textrank.extract_sentences() is used to get summary of the file

textrank.extract_key_phrases() is used to get keywords of the file

## Abstractive (Sprint 3)
Generate a summary that keeps original intent. It's just like humans do.Based on the paper,“ A Review on Automatic Text Summarization Approaches ” https://thescipub.com/PDF/jcssp.2016.178.190.pdf，we choose to use RNN and CNN as our model.The code is in …,for details of the codes,you can read this paper.https://towardsdatascience.com/text-summarization-with-amazon-reviews-41801c2210b

