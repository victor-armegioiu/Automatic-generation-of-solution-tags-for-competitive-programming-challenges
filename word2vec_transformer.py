from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

import re
import matplotlib.pyplot as plt
import pandas as pd

import os
import numpy as np

from nltk_utils import get_lemmatized_tokens
from nltk_utils import clean

from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))

if __name__ == '__main__':
	data = pd.read_csv('codeforces_problems_csv/data.csv')
	X_data = list(data['problem_text'])

	if not os.path.exists('w2v_problem_data.bin'):
		sentences = [line for text in X_data for line in clean(text)]
		#for i in range(len(sentences)):
		#	sentences[i] = get_lemmatized_tokens(' '.join(sentences[i]))

		model = Word2Vec(sentences, workers=4, size=200, min_count=50, window=10, sample=1e-3)
		model.save('w2v_problem_data.bin')

	else:
		model = Word2Vec.load('w2v_problem_data.bin')

	# very common word in dp problems
	print(model.most_similar('ways'))
	print(len(model.wv.vocab.values()))

	X = model[model.wv.vocab]

	# visualize the data
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(X)

	plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
	plt.show()