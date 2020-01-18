from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
output = 1
# App config.
DEBUG = True
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import time
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
RS = 123
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
from scipy.spatial.distance import cosine

# def get_data():
# 	"""
# 	Gets data from online source and saves it in Pandas DataFrame
# 	"""
# 	page = requests.get("http://www.grecoaching.com/allalphabets.html")
# 	soup = BeautifulSoup(page.content, 'html.parser')
# 	output = [tag.text for k in range(3) for tag in soup.find(class_ = "wordlist2").select('ul')[k].find_all('li', recursive = True)]
# 	for i in [77,223,395,510]:
# 		if i != 510:
# 			output.pop(i)
# 		else:
# 			for j in range(8):
# 				output.pop(i)
# 	temp = []
# 	for def_ in output:
# 		temp.append((def_.split(' - ')[0], def_.split(' - ')[2].split(";")[0]))
# 	data = pd.DataFrame(temp, columns = ['word', 'definition'])
# 	return data

# def tokenize(data):
# 	"""
# 	Tokenizes the data definition
# 	"""
# 	tokenized = []
# 	to_remove = []
# 	count = 0
# 	for def_ in data.definition:
# 		doc = nlp(def_.lower())
# 		tokens = [token.text for token in doc if (not token.is_stop and not token.is_punct and token.text.strip() is not u'')]
# 		tokenized.append(tokens)
# 		if len(tokens)<=0:
# 			to_remove.append(count)
# 			count += 1
# 	tokenized = [v for i,v in enumerate(tokenized) if i not in to_remove]
# 	return tokenized, to_remove

# def vectorize(tokenized):
# 	"""
# 	Finds average word embeddings of the definition
# 	"""
# 	vectors = []
# 	for bow in tokenized:
# 		op = np.zeros(300)
# 		count = 0
# 		for word in bow:
# 			try:
# 				op += nlp(word.lower()).vector
# 				count += 1
# 			except:
# 				pass
# 			if count != 0:
# 				op /= count
# 		vectors.append(op)
# 	return vectors

# def set_search_space(vectors, data, to_remove):
# 	"""
# 	Divides the words into 3 distinct clusters
# 	"""
# 	clusters = KMeans(n_clusters=3, max_iter = 1000, random_state = RS).fit(vectors).labels_ 
# 	search_space_0 = []
# 	search_space_1 = []
# 	search_space_2 = []
# 	cluster_0_avg = np.zeros(300)
# 	cluster_1_avg = np.zeros(300)
# 	cluster_2_avg = np.zeros(300)
# 	words = list(data.word)
# 	words = [v for i,v in enumerate(words) if i not in to_remove]
# 	clusters = list(clusters)
# 	for i in range(len(clusters)):
# 		if clusters[i] == 0:
# 			# print cluster_0_avg
# 			search_space_0.append((words[i], vectors[i]))
# 			cluster_0_avg += vectors[i]
# 	# print cluster_0_avg
# 		if clusters[i] == 1:
# 			# print cluster_1_avg
# 			search_space_1.append((words[i], vectors[i]))
# 			cluster_1_avg += vectors[i]
# 	# print vectors[i]
# 		if clusters[i] == 2:
# 			search_space_2.append((words[i], vectors[i]))
# 			cluster_2_avg += vectors[i]
# 	cluster_0_avg, cluster_1_avg, cluster_2_avg = cluster_0_avg/float(len(search_space_0)), cluster_1_avg/float(len(search_space_1)), cluster_2_avg/float(len(search_space_2))
	
# 	return clusters, [search_space_0, search_space_1, search_space_2], [cluster_0_avg, cluster_1_avg, cluster_2_avg]

# def get_input_vector():
# 	"""
# 	Gets input from the user and converts it into average word embeddings
# 	"""
# 	input = raw_input()
# 	input = unicode(input,"utf-8")
# 	input_tokens = []
# 	input_vector = np.zeros(300)
# 	count = 0
# 	doc = nlp(input.lower())
# 	tokens = [token.lemma_ for token in doc if (not token.is_stop and not token.is_punct and token.text.strip() is not u'')]
# 	input_tokens.append(tokens)
# 	for word in input_tokens[0]:
# 		try:
# 			input_vector += nlp(word).vector
# 			count += 1
# 		except:
# 			pass
# 		if count != 0:
# 			input_vector /= count
# 	return input_vector

# def search_space_selector(input_vector, clusters_avg):
# 	"""
# 	Reduces search space to a specific cluster
# 	"""
# 	sss = np.zeros(3)
# 	for i in range(3):
# 		sss[i] = 1-cosine(input_vector.reshape(-1,1), clusters_avg[i].reshape(-1,1))
# 	return sss.argmax()

# def word_selector(input_vector, search_space, search_space_selector):
# 	"""
# 	Selects the top five words that matches the query in the specific cluster
# 	"""
# 	ws = np.zeros(len(search_space[search_space_selector]))
# 	count = 0
# 	for word, word2vec in search_space[search_space_selector]:
# 		ws[count] = 1- cosine(input_vector.reshape(-1,1), word2vec.reshape(-1,1))
# 		count += 1
# 	return ws.argsort()[-5:]

# def get_word_match(word_selector,search_space_selector, search_space):
# 	"""
# 	Returns the top five matches for the query
# 	"""
# 	best_match = []
# 	for i in word_selector.argsort()[-5:]:
# 		best_match.append(search_space[search_space_selector][i][0])
# 	return best_match

app = Flask(__name__)
@app.route("/")
def hello():

    return render_template('index.html')

if __name__ == "__main__":
    app.run()
# df = get_data()
# tokenized, remove = tokenize(df)
# vectors = vectorize(tokenized)
# clusters, search_space, clusters_avg = set_search_space(vectors, df, remove)

# iv = get_input_vector()
# sss = search_space_selector(iv, clusters_avg)
# ws = word_selector(iv, search_space, sss)
# best_match = get_word_match(ws,sss, search_space)

# print ", ".join(best_match)