import math
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
from sklearn.metrics.pairwise import cosine_similarity
import random
#Task T1
#Read Input file & tokenize it
file_content = open("/Users/macbook/Documents/Computing lab/gopal/Mtech CSE Sem 1/CL/Networkx and NLTK/input.txt").read()
f = open("Summary_SentenceGraph.txt", "w")
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

#print(sentence_x_words_matrix)
#find page rank of graph
pr=nx.pagerank(G,0.4)
#print(pr)
#sort on pagerank and print the top n sentences in proper sequence
Sorted_on_page_rank = dict(sorted(pr.items(), key = lambda x: x[1], reverse = True))


#Using MMR to omit ‘redundancy’ in the summary

def MMR(lamda , number_of_summary_sent):
	Selected_set = [[x,y] for x,y in Sorted_on_page_rank.items()]
	Selected_set = [Selected_set[0]]
	unselected_set = [[x,y] for x,y in Sorted_on_page_rank.items()][1:]
	while(number_of_summary_sent):
		max_rank =-1
		max_rank_index=-1
		for s in range(0,len(unselected_set)) :
			s_max = -1
			s_max_index=-1
			for x in range(0,len(Selected_set)):
				sim = 1-cosine_distance(sentence_x_words_matrix[Selected_set[x][0]],sentence_x_words_matrix[unselected_set[s][0]])
				if(sim>s_max):
					s_max = sim
					s_max_index= x
			ranking_score = lamda*(unselected_set[s][1]) - (1-lamda)*(s_max)			
			unselected_set[s][1] = ranking_score
			if(max_rank<ranking_score):
				max_rank=ranking_score
				max_rank_index = s
		Selected_set.append(unselected_set[max_rank_index])
		unselected_set.pop(max_rank_index)
		number_of_summary_sent-=1
	return  Selected_set

MMM_final = MMR(0.5,1)
MMM_final.sort(key = lambda x: x[0])
print("\nMMR Summary :-")
f.write("\nMMR Summary :-\n\n")
for i in MMM_final:
	f.writelines([sentences[i[0]],"\n"])
	print(sentences[i[0]])
	
#Clustering and Word-Graph approach
#T1. Sentence clustering using K-Means
#1. Choose the number of clusters(K)
no_of_clusters = 1
#2. Place the centroids c_1, c_2, ..... c_k randomly
#(use random.sample utility in python for this)
random.seed(42)
clusters = random.sample([[i] for i in range(0,total_no_of_sentences)],no_of_clusters)
centroids = [sentence_x_words_matrix[i[0]] for i in clusters]
#3. Repeat steps 4 and 5 until convergence — cluster assignment does
#not change
while(True):
    flag=0
    for i in range(0,total_no_of_sentences):
        #4. for each sentence i:
        #- find the nearest centroid(c_1, c_2 .. c_k) —using cosine similarity on tf-idf representation
        nearest_centroid = -1
        max_similarity = -1
        for j in range(0,no_of_clusters):
            similar_score =1-cosine_distance(sentence_x_words_matrix[i],centroids[j])
            if(max_similarity<similar_score):
                max_similarity=similar_score
                nearest_centroid=j
        if(i not in clusters[nearest_centroid]):
            #- assign the sentence to that cluster
            clusters[nearest_centroid].append(i)
        for p in range(0,no_of_clusters):
            if(p!=nearest_centroid and i in clusters[p]):
                flag=1
                clusters[p].remove(i)
    #5. for each cluster j = 1..k
    #- new centroid = mean of all points in that cluster
    for i in range(0,no_of_clusters):
      sum = sentence_x_words_matrix[clusters[i][0]]
      for j in range(1,len(clusters[i])):
        sum = [sum[x]+ sentence_x_words_matrix[clusters[i][j]][x] for x in range(0,len(sum))]
      new_mean = [sum[x]/len(clusters[i]) for x in range(0,len(sum))]
      centroids[i] = new_mean
    if (flag==0):
      break

sentences = [i.lower() for i in sentences]

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)
#T2. Obtaining ‘new’ sentences using Sentence Graph
final_sent_of_each_cluster = {}
#1. For each of the clusters,
for i in range(0,no_of_clusters):
    #a. identify the sentence that is closest to the cluster centroid. In case of ties, pick any one at
    #random — S1
    S1=-1
    max_similarity = -1
    for c in range(0,len(clusters[i])):
        similar_score =1-cosine_distance(sentence_x_words_matrix[clusters[i][c]],centroids[i])
        if(max_similarity<similar_score):
            max_similarity=similar_score
            S1=clusters[i][c]
    
    bigram_S1 = [(x, sentences[S1].split()[j + 1]) for j, x in enumerate(sentences[S1].split()) if j < len(sentences[S1].split()) - 1]
    #. Find a sentence in the cluster that has at least 3 bigrams in common with S1 — S2.
    S2=-1
    count_max=0
    bigram_S2_final = list()
    for c in range(0,len(clusters[i])):
        if(clusters[i][c]!=S1):
            bigram_S2 = [(x, sentences[clusters[i][c]].split()[j + 1]) for j, x in enumerate(sentences[clusters[i][c]].split()) if j < len(sentences[clusters[i][c]].split()) - 1]
            count = len(Intersection(bigram_S2,bigram_S1))
            if(count>=3 and count_max<count):
                count_max=count
                S2=clusters[i][c]
                bigram_S2_final=bigram_S2
    #c. In case no S2 exists for a cluster — save only S1 from this step and return. Do not continue with Step d. and part 2.
    if(S2==-1):
        final_sent_of_each_cluster[S1] = sentences[S1]
    else:
        G_sent = nx.DiGraph()
        S1_len =len(bigram_S1)
        S2_len =len(bigram_S2_final)
        G_sent.add_nodes_from(["start","end",bigram_S1[0],bigram_S1[S1_len-1],bigram_S2_final[0],bigram_S2_final[S2_len-1]])
        G_sent.add_edge("start",bigram_S1[0])
        G_sent.add_edge("start",bigram_S2_final[0])
        G_sent.add_edge(bigram_S1[S1_len-1],"end")
        G_sent.add_edge(bigram_S2_final[S2_len-1],"end")
        for j in range(1,len(bigram_S1)-1):
            G_sent.add_node(bigram_S1[j])
            G_sent.add_edge(bigram_S1[j-1],bigram_S1[j])
        for j in range(1,len(bigram_S2_final)-1):
            if(not G.has_node(bigram_S2_final[j])):
                G_sent.add_node(bigram_S2_final[j])
            G_sent.add_edge(bigram_S2_final[j-1],bigram_S2_final[j])
        G_sent.add_edge(bigram_S2_final[S2_len-2],bigram_S2_final[S2_len-1])
        G_sent.add_edge(bigram_S1[S1_len-2],bigram_S1[S1_len-1])
        final_of_two = list(nx.all_simple_paths(G_sent, "start", "end"))
#        pos = nx.spectral_layout(G_sent)
#        nx.draw(G_sent,pos,with_labels=True,node_size=1000,connectionstyle='arc3, rad = 0.1')
#        print(final_of_two)
        final_sent_of_each_cluster[S1] = final_of_two[random.sample([i in range(0,len(final_of_two))],1)[0]] # choose random here
        #plt.show()
final_sent_of_each_cluster = list(final_sent_of_each_cluster.items())
final_sent_of_each_cluster.sort(key = lambda x: x[0])


print("\nFinal Summary :-")
f.write("\nFinal Summary :-\n\n")
for i in final_sent_of_each_cluster:
  sent_len = len(i[1])
  if(type(i[1])==type('str')):
      print(i[1],end=".")
      f.write(i[1]+".")
      continue
  for j in range(1,sent_len-1):
    print(i[1][j][0],end=" ")
    f.write(i[1][j][0]+" ")
  print(i[1][j][1],end=" ")
  f.write(i[1][j][1]+" ")
  print(".",end=" ")
  f.write(". ")
f.close()
