import numpy as np
import pandas as pd
from os import path, listdir
import PIL
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax


def importData(path_dir, delimeter):
	data = []
	for filename in listdir(path_dir):
		t = pd.read_csv(path.join(path_dir,filename), delimiter=delimeter)
		data.append(t)
	return pd.concat(data)

def drawPlot(title, df, xlabel, ylabel):
	plt.figure(figsize=(15,10))
	df.size().sort_values(ascending=False).plot.bar()
	plt.xticks(rotation=50)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

def drawWordCloud(text):
	wordcloud = WordCloud(max_words=100).generate(text)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()

def createVocab(doc):
	fdist = FreqDist()
	for i in doc:
		for j in sent_tokenize(i):
			for word in word_tokenize(j):
				fdist[word.lower()] += 1
	return fdist

def OneHoteDecode(label_encoder, doc):
    inverted = label_encoder.inverse_transform([argmax(doc)])
    return inverted
	

def oneHotEncode(doc):
	# integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(doc)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return label_encoder, onehot_encoded
	

def removeHashTag(doc, column):
	doc[column] = doc.apply(lambda x: re.sub(r'(\#[a-zA-Z0-9\_]*)', "", x[column]), axis=1)
	return doc

def removeNumberAndSymbol(doc, column):
	doc[column] = doc.apply(lambda x: re.sub(r'([\_\*\~\=\+\#\-\:\…\(\)\[\]\.0-9]+)', " ", x[column]), axis=1)
	return doc

def removeamp(doc, column):
	doc[column] = doc.apply(lambda x: re.sub(r'(&amp;)|(\\n)|(\\t)|(\W\'\W)', "", x[column]), axis=1)
	return doc
def removeUnicode(doc, column):
	doc[column] = doc.apply(lambda x: x[column].encode('ascii', errors='ignore').strip().decode('ascii'), axis=1)
	return doc

def removeMention(doc, column):
	doc[column] = doc.apply(lambda x: re.sub(r'((RT\s*)*\@[a-zA-Z0-9\_]+(\s*\:)*)', "", x[column]), axis=1)
	return doc

def removeLink(doc, column):
	doc[column] = doc.apply(lambda x: re.sub(r'(http[a-zA-Z0-9\\\-\:\/\.]+)', "", x[column]), axis=1)
	return doc