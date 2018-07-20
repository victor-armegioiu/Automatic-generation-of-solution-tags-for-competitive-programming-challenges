import os
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
import sys


part = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}


def clean(text):
    lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '', re.sub('\n', ' ', text)))
    return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]


def get_wordnet_pos(treebank_tag):
    first_letter = treebank_tag[0]
    return part[first_letter] if first_letter in part else 'n'


def lemmatize_tokens(tokens):
	lemmatizer = WordNetLemmatizer()
	token_tags = nltk.pos_tag(tokens)

	lemmatized_tokens = []
	for token, token_tag in token_tags:
		lemmatized_tokens.append(lemmatizer.lemmatize(token, get_wordnet_pos(token_tag)))

	return lemmatized_tokens


def get_problem_texts():
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	files.remove('topics.py')
	files.remove('out')
	files.remove('t.py')

	documents = []

	for file_name in files:
		lines = open(file_name).read().splitlines()

		lines = [line.rstrip() for line in lines[2:]]
		lines = list(filter(lambda x : len(x), lines))

		lines = ' '.join(lines)
		lines = list(map(str.lower, word_tokenize(lines)))

		lines = lemmatize_tokens(lines)
		lines = ' '.join([line for line in lines if line.isalpha()])

		documents.append(lines)
		
	return documents

def get_lemmatized_tokens(problem_text):
	problem_text = list(map(str.lower, word_tokenize(problem_text)))
	problem_text = lemmatize_tokens(problem_text)
	problem_text = list(filter(str.isalpha, problem_text))

	return problem_text


def valid(word):
	return len(word) > 1 and word not in ['one', 'two']


if __name__ == '__main__':
	documents = get_problem_texts()
	word_list = set(words.words())

	stoplist = set(stopwords.words('english'))
	texts = [[word for word in document.lower().split() 
			if word not in stoplist and word in word_list
			and valid(word)]
	         for document in documents]

	for i in range(len(texts)):
		texts[i] = ['_'.join(w) for w in ngrams(texts[i], 2)]

	for text in texts:
		print(text)
		print(100 * '_')
	# Create Dictionary.
	id2word = corpora.Dictionary(texts)
	
	# Creates the Bag of Word corpus.
	mm = [id2word.doc2bow(text) for text in texts]

	# Trains the LDA models.
	lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=1, \
	                               update_every=1, chunksize=10000, passes=1,
	                               alpha=0.05)

	# Prints the topics.
	for top in lda.print_topics():
	  print(top)
	print()
