import multiprocessing
from gensim.models import Word2Vec
from time import time

import logging # Setting upp the loggings to monitor gensim
logging.basicConfig(filename="Logger(1).log", format="%(levelname)s - %(asctime)s: %(message)s", datefmt = '%H:%M:%S', level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('darkgrid')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Read sentences from file
logging.info("Start load sentences from file")
fname = "simpsons_trainingset.txt"
sentences = []
with open(fname, 'r', encoding='utf-8') as f:
    while True:
        sent = f.readline()
        if len(sent) == 0:
            break
        sent = sent[:-1].split()
        sentences.append(sent)

logging.info("Count the number of cores in this computer")
cores = multiprocessing.cpu_count() # Count the number of cores in a computer

# Train
min_count = 20
window = 2
size = 300
sample = 6e-5
alpha = 0.03
min_alpha = 0.0007
negative = 20
hs = 0
sg = 0
logging.info("Start training with min_count={0}, window={1}, size={2}, sample={3}, alpha={4}, min_alpha={5}, negative={6}. sg={7}, hs={8}"
             .format(min_count, window, size, sample, alpha, min_alpha, negative, sg, hs))
w2v_model = Word2Vec(min_count=min_count,
                     hs=hs,
                     window=window,
                     size=size,
                     sample=sample,
                     alpha=alpha,
                     min_alpha=min_alpha,
                     negative=negative,
                     workers=cores-1)

t = time()

w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round(time() - t) / 60, 2))

t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round(time() - t) / 60, 2))

w2v_model.init_sims(replace=True)

# Exploring the model
print("Cac tu gan voi tu 'homer': ")
print(w2v_model.wv.most_similar(positive=['homer']))
print()
print("Cac tu gan voi tu 'homer_simpson': ")
print(w2v_model.wv.most_similar(positive=["homer_simpson"]))
print()
print("Cac tu gan voi tu 'marge': ")
print(w2v_model.wv.most_similar(positive=["marge"]))
print()

print("Similarity giua 'maggie' va 'baby': ")
print(w2v_model.wv.similarity('maggie', 'baby'))
print()

print("Tu khong thuoc trong ba tu 'jimbo', 'milhouse', 'kearney': ")
print(w2v_model.wv.doesnt_match(['jimbo', 'milhouse', 'kearney']))
print()

print("Which word is to 'woman' as 'homer' is to 'marge'? ")
print(w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3))
print()

print("Which word is to 'woman' as 'bart' is to 'man'? ")
print(w2v_model.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3))
print()

logging.info("\n")

# T-SNE
def tsnescatterplot(model, word, list_names):
    """Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words."""
    
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
    print(arrays.shape)
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.show()

tsnescatterplot(w2v_model, 'homer', ['dog', 'bird', 'ah', 'maude', 'bob', 'mel', 'apu', 'duff'])
