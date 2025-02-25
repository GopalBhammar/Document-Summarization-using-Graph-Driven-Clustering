# Document-Summarization-using-Graph-Driven-Clustering
Text Summarization Using NLTK and NetworkX

# 📜 Text Summarization Using NLTK and NetworkX  

## 📌 Overview  
This project implements various **text summarization techniques** using **NLTK** and **NetworkX**. It processes a document, applies **NLP techniques**, and extracts key sentences using:  
- **TF-IDF**
- **PageRank**
- **Maximal Marginal Relevance (MMR)**
- **Sentence Clustering Approach**  

## 🚀 Features  

### 🔹 **Extractive Summarization with PageRank**  
#### 📝 Preprocessing (NLTK)  
- Sentence segmentation  
- Removing punctuation & stopwords  
- Tokenization & lemmatization  

#### 📊 Sentence Representation (TF-IDF)  
- Compute **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**  
- Construct a **TF-IDF matrix**  

#### 🔗 Graph-Based Summarization (PageRank)  
- Construct a **sentence similarity graph**  
- Compute **PageRank** to rank sentences  
- Generate summary (**stored in `Summary_PR.txt`**)  

---

### 🔹 **Reducing Redundancy in Summarization**  

#### 🎯 Maximal Marginal Relevance (MMR)  
- Rerank sentences to **introduce diversity**  
- Eliminate **redundant sentences** in the summary  

#### 📌 Clustering-Based Summarization  
- Apply **K-Means clustering** to group similar sentences  
- Select **representative sentences** from each cluster  

#### 🔗 Word Graph-Based Sentence Fusion  
- Combine similar sentences to generate **new meaningful sentences**  

#### 📝 Final Ordered Summary  
- Arrange sentences based on their **positions in the original document**  
- Store output in **`Summary_SentenceGraph.txt`**  

---

## 🛠 Installation & Usage  

### 🔹 **Requirements**  
To install dependencies, run:  
```bash
pip install nltk networkx scikit-learn


