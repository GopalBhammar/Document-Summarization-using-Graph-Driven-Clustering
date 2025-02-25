import math
import nltk
from nltk.corpus import stopwords
# print(stopwords.words('english'))
import nltk
from functools import reduce
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import LineTokenizer
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from nltk.cluster.util import cosine_distance
#Task T1
#Read Input file & tokenize it
file_content = open("/Users/macbook/Documents/Computing lab/gopal/Mtech CSE Sem 1/CL/Networkx and NLTK/input.txt").read()
tk = LineTokenizer()
sentences = tk.tokenize(file_content)

#Remove punctuation
sentences = [i.translate (str.maketrans ('', '', string.punctuation)) for i in sentences ]

#revome stopwords and apply Lemmatize
english_stopwords = set(stopwords.words('english'))
ps = WordNetLemmatizer()
final_sentences=[]
for s in sentences:
	words = word_tokenize(s)
	filtered_words = [word for word in words if word.lower() not in english_stopwords]
	final_sentences.append(reduce(lambda x, y: x + " " + ps.lemmatize(y), filtered_words, ""))
#print(final_sentences)


#Task T2
total_no_of_sentences = len(final_sentences)
words_all_sentences = set()
for s in final_sentences:
	words = word_tokenize(s)
	words_all_sentences = words_all_sentences.union(words)

#Compute TF-Idf Matrix
tf_idf_matrix = []
for word in words_all_sentences:
	row = list()
	no_of_sent_with_w = 0
	for sentence in final_sentences:
		words = word_tokenize(sentence)
		tf = Counter(words)
		row.append(tf[word])
		if(tf[word]) :
			no_of_sent_with_w+=1
	idf = math.log(total_no_of_sentences/no_of_sent_with_w)
	row=[x*idf for x in row]
	tf_idf_matrix.append(row)
#cc=0
#for i in words_all_sentences:
#	print(i,tf_idf_matrix[cc])
#	cc+=1
#Task T3
#creating graph
G = nx.DiGraph()
nodes = [ x for x in range(0,total_no_of_sentences)]
G.add_nodes_from(nodes)

#Compute cosine for weigth in edge
#		cosine = np.dot(sentence_x_words_matrix[i],sentence_x_words_matrix[j])
#		cosine /= (norm(sentence_x_words_matrix[i])*norm(sentence_x_words_matrix[j]))
sentence_x_words_matrix =  np.transpose(tf_idf_matrix)
for i in range(0,total_no_of_sentences):
	for j in range(0,total_no_of_sentences):
		G.add_edge(i,j,weight=(1-cosine_distance(sentence_x_words_matrix[i],sentence_x_words_matrix[j])))


#find page rank of graph
pr=nx.pagerank(G,0.4)
#print(pr)
#sort on pagerank and print the top n sentences in proper sequence
Sorted_on_page_rank = dict(sorted(pr.items(), key = lambda x: x[1], reverse = True))

def Top_n (n):
	top = list(Sorted_on_page_rank.items())[:n]
	return top

f = open("Summary_PR.txt", "w")
result= Top_n(7)
result.sort(key = lambda x: x[0])
print("Summary of above sentences is : ")
for i in result:
	f.writelines([sentences[i[0]],"\n"])
	print(sentences[i[0]])
