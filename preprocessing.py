import re #For preoprocessing
import pandas as pd # For data handling
from time import  time # To time our operations
from collections import  defaultdict # For word frequency

import spacy # For preprocessing

import logging # Setting upp the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt = '%H:%M:%S', level=logging.INFO)

from gensim.models.phrases import Phrases, Phraser


#Preprocessing
df = pd.read_csv('simpsons_dataset.csv')
#print("Shape:",df.shape)
#df.isnull().sum()
df = df.dropna().reset_index(drop=True)
#df.isnull().sum()

#Cleaning
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def cleaning(doc):
    #Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word.
    # if a sentences only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return " ".join(txt)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {0} mins'.format(round((time() - t) / 60, 2)))

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
#print(df_clean.shape)

#Bigrams

sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent, min_count=30, progress_per= 10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

# Save to
fname = "simpsons_trainingset.txt"

with open(fname,'w',encoding = 'utf-8') as f:
    for sent in sentences:
        for i in sent:
            f.write("%s " % i)
        f.write('\n')

"""print(len(sentences))
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
print(len(word_freq))
sorted(word_freq, key=word_freq.get, reverse=True)[:10]"""
