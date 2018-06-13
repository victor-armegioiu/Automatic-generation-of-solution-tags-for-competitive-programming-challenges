from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from skmultilearn.problem_transform import LabelPowerset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier

from skmultilearn.neurofuzzy import MLARAM
from skmultilearn.adapt.brknn import BRkNNaClassifier
from sklearn.naive_bayes import GaussianNB
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import ClassifierChain

from xgboost import XGBClassifier

import tensorflow_hub as hub
import tensorflow as tf

import numpy as np
import pandas as pd

from scipy import sparse
from nltk.corpus import stopwords
from nltk_utils import clean

stoplist = set(stopwords.words('english'))

MODEL = 0
EMBEDDING_TYPE = 1
REQUIRES_DENSE = 2

def universal_embeddings(X):
	embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
	embeddings = embed(X)

	toks_embeddings = None

	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
		toks_embeddings = sess.run(embeddings)

	return toks_embeddings


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 3000

    def fit(self, X, y):
        return self

    def transform(self, X):
    	x_tr = []
    	for i in range(len(X)):
	    	tokenized = [item for sublist in clean(str(X[i])) for item in sublist]
	    	tokenizer = list(set(tokenized) - stoplist)

	    	curr_tr = np.zeros(self.dim)

	    	for j in range(len(tokenized)):
	    		curr_tr[j] = np.mean(self.word2vec.wv[tokenized[j]]) if tokenized[j] in self.word2vec.wv else 0

	    	x_tr.append(curr_tr)
    	return sparse.csr_matrix(np.array(x_tr))


class UniversalSentenceEmbeddings(object):
	def __init__(self):
		pass

	def fit(self, X, y):
		return self

	def transform(self, X):
		return universal_embeddings(X)


class DenseTransformer():
	    def transform(self, X, y=None, **fit_params):
	        return X.todense()

	    def fit_transform(self, X, y=None, **fit_params):
	        self.fit(X, y, **fit_params)
	        return self.transform(X)

	    def fit(self, X, y=None, **fit_params):
	        return self


def sklearn_evaluator(X_train, y_train, X_test, y_test, models):
	for model_name in models:
		pipeline_steps = []
		pipeline_steps.append(('vectorizer', models[model_name][EMBEDDING_TYPE]))

		if models[model_name][REQUIRES_DENSE]:
			pipeline_steps.append(('to-dense', DenseTransformer()))

		pipeline_steps.append(('clf', models[model_name][MODEL]))

		classifier = Pipeline(pipeline_steps)
		classifier.fit(X_train, y_train)

		y_pred = classifier.predict(X_test)

		if model_name == 'BRkNN':
			y_pred = y_pred.toarray()
			y_pred[y_pred > 1] = 1

		f1_accuracy = f1_score(y_test, y_pred, average='micro')
		print(model_name, ':', f1_accuracy)


if __name__ == '__main__':
	
	data = pd.read_csv('codeforces_problems_csv/data.csv')
	X_data = data['problem_text']
	y_data = []

	unprocessed = data['tags']
	for i in range (len(unprocessed)):
		unprocessed[i] = list(map(int, unprocessed[i][1:-1].split(', ')))
		y_data.append(unprocessed[i])

	X_data = np.array(X_data)
	y_data = np.array(y_data)

	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
										test_size=0.23, random_state=42)


	one_vs_rest_sklearn_models = {
									'Linear SVM' : (OneVsRestClassifier(LinearSVC()), TfidfVectorizer(stop_words={'english'}), False),
									'Knn' : (OneVsRestClassifier(KNeighborsClassifier()), TfidfVectorizer(stop_words={'english'}), False),
									'GradientBoostingClassifier' : (OneVsRestClassifier(GradientBoostingClassifier()), TfidfVectorizer(stop_words={'english'}), False),
									'XGBoost' : (OneVsRestClassifier(XGBClassifier()), TfidfVectorizer(stop_words={'english'}), False)
								}


	multi_label_sklearn_models = {
									'ML-ARAM' : (MLARAM(), TfidfVectorizer(stop_words={'english'}), True),
									'BRkNN' : (BRkNNaClassifier(), TfidfVectorizer(stop_words={'english'}), False),
									'RandomForestClassifier' : (RandomForestClassifier(), TfidfVectorizer(stop_words={'english'}), False),
									'MLkNN' : (MLkNN(k=20), TfidfVectorizer(stop_words={'english'}), False),
									'Classifier Chain' : (ClassifierChain(LinearSVC()), TfidfVectorizer(stop_words={'english'}), False),
									'Label Powerset' : (LabelPowerset(LinearSVC()), TfidfVectorizer(stop_words={'english'}), False)
								}

	sklearn_evaluator(X_train, y_train, X_test, y_test, multi_label_sklearn_models)
	#sklearn_evaluator(X_train, y_train, X_test, y_test, one_vs_rest_sklearn_models)

	# really bad results with pretrained word2vec
	"""
	from gensim.models.word2vec import Word2Vec
	import gensim
	#w2vmodel = Word2Vec.load('w2v_problem_data.bin')
	w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
	classifier = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2vmodel)),("SVM", OneVsRestClassifier(LinearSVC()))])

	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	f1_accuracy = f1_score(y_test, y_pred, average='micro')

	print('W2V acc: ', f1_accuracy)
	"""	