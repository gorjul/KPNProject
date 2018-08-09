## put all imported modules here
from collections import Counter
import pandas as pd
import re
from nltk.tokenize import word_tokenize
#from nltk.stem.snowball import SnowballStemmer
# from gensim.models.tfidfmodel import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
#other imports
from os import listdir
from nltk.tokenize import RegexpTokenizer
import gensim


################# 1. IMPORT DATA ################################################################
# Import the csv file
data = pd.read_csv(r'C:\Users\mestr501\Desktop\Thesis\Python\textcsv.csv', 
                   delimiter = ';', header = 0, encoding = 'latin-1') #latin-1 for dutch

# Turn whole_file column into document or txt
whole_files = data['whole_file']
corpus = whole_files.tolist()
print(corpus)

# Turn whole_file column into document or txt
client_names = data['client_name']
client_names = client_names.tolist()
print(client_names)

# The stopword list and company names added to them
sw = []
with open(r'C:\Users\mestr501\Desktop\Thesis\Python\stopwords-nl.txt', encoding = 'utf-8') as handle:
    for line in handle:
        sw.append(line.strip())
print(sw)
sw.append('kpn') #if more stopwords, add more appends
sw.append('én')
#for term in client_names: #adding client names to stopword list.
#    term = term.lower()
#    sw.append(term)



################ 2. PRE-PROCESSING STEPS ########################################################
# 2.1 Tokenize the text
# Tokenize the cells per word
tknzd_corpus = []
for doc in corpus:
    tokens = word_tokenize(doc) 
    tknzd_corpus.append(tokens)

print(tknzd_corpus)

# 2.2 Lower all cases
################ corpus = [t.lower() for t in tknzd_corpus]
tknzd_corpus_lower = []
for doc in tknzd_corpus:
    tknzd_corpus_lower.append([t.lower() for t in doc])
print(tknzd_corpus_lower[0])

###### 2.3 Delete punctuation, numbers, stopwords, empty spaces and stem if needed
punctuation = r'[^\w\s]'
digits = r"\d+"
#stemmer = SnowballStemmer("dutch")

words = []
for doc in tknzd_corpus_lower :
    doc = [re.sub(digits, '', w) for w in doc]          # removal of digits
    doc = [t for t in doc if t not in sw]               # removal of stop words
    doc = [re.sub(punctuation, '', t) for t in doc]     # removal of punctuation
    #doc = [stemmer.stem(t) for t in doc]               # stemming the words
    doc = [t for t in doc if t!='']                     # delete empty spaces/elements
    words.append(doc) 
print(words[0:2])



################ 3. TF IDF Vector creation ####################################################
from gensim import corpora, models
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(words)
print(dictionary) 
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(words) for word in words]

#create TF IDF
tfidf = models.TfidfModel(corpus)

#use model to transform corpus
corpus_tfidf = tfidf[corpus]

#create LSA 
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)

#tfidf_vectorizer = TfidfVectorizer(input = cleaned_text, 
#                                   encoding ='utf-8', 
#                                   strip_accents = 'ascii',
#                                   analyzer = 'word',
#                                   max_df = 0.8, min_df = 0.1,
#                                   use_idf = True)
#tfidf_vectorizer
#
##
#tfidf_train = tfidf_vectorizer.fit_transform(cleaned_text) 
##
### Transform the test data: tfidf_test 
##tfidf_test = tfidf_vectorizer.transform(cleaned_text)
#
## Print the first 10 features
#print(tfidf_vectorizer.get_feature_names())

################## 4. SVD CREATION ############################################################
#n_components is recommended to be 100 by Sklearn Documentation for LSA
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
svd = TruncatedSVD(n_components = 100,
                   algorithm = 'randomized',
                   n_iter = 10,              #number of iterations
                   random_state = 123)      #set as a seed

#creating SVD matrix
svd_transformer = Pipeline([('tfidf', tfidf_vectorizer), 
                            ('svd', svd)])
svd_matrix = svd_transformer.fit_transform(cleaned_text)
print(svd_transformer.get_feature_names()[:10])

################## 5. COSINE DISTANCE #################################################

    
    
    