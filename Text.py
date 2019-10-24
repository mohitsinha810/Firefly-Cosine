import numpy

from nlp_utils import *

import math
import numpy as np

from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def get_all_content_words_lemmatized(sentences, N=1):
    all_words = []
    for s in sentences:
        all_words.extend([wordnet_lemmatizer.lemmatize(r) for r in tokenizer.tokenize(s)])
    if N == 1:
        content_words = [w for w in all_words if w not in stopset]
    normalized_content_words = map(normalize_word, content_words)
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N)]
    return normalized_content_words


def get_all_content_words_stemmed(sentences, N=1):
    def is_ngram_content(g):
        for a in g:
            if not (a in stopset):
                return True
        return False

    all_words = []
    for s in sentences:
        all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])

    if N == 1:
        content_words = [w for w in all_words if w not in stopset]
    else:
        content_words = all_words

    normalized_content_words = map(normalize_word, content_words)
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N) if is_ngram_content(gram)]
    return normalized_content_words


def get_all_content_words(sentences, N=1):
    all_words = []
    for s in sentences:
        all_words.extend(tokenizer.tokenize(s))
    content_words = [w for w in all_words if w not in stopset]
    normalized_content_words = map(normalize_word, content_words)
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N)]
    return normalized_content_words


def get_content_words_in_sentence(sentence):
    words = tokenizer.tokenize(sentence)
    return [w for w in words if w not in stopset]


def kl_divergence(summary_freq, doc_freq):
    sum_val = 0
    for w, f in summary_freq.items():
        if w in doc_freq:
            sum_val += f * math.log(f / float(doc_freq[w]))

    return sum_val


def compute_tf_doc(docs, N=1):
    sentences = []
    for title, doc in docs:
        sentences.append(title)
        sentences.extend(doc)

    content_words = list(set(get_all_content_words_stemmed(sentences, N)))
    docs_words = []
    for title, doc in docs:
        s_tmp = [title]
        s_tmp.extend(doc)
        docs_words.append(get_all_content_words_stemmed(s_tmp, N))

    word_freq = {}
    for w in content_words:
        w_score = 0
        for d in docs_words:
            if w in d:
                w_score += 1
        if w_score != 0:
            word_freq[w] = w_score

    content_word_tf = dict((w, f / float(len(word_freq.keys()))) for w, f in word_freq.items())
    return content_word_tf


# Computes (no. of occurrences of word in a document/Total no. of words in document)
# This is used to calculate the tf values for generating the tf-idf matrix
def compute_word_freq(words, all_words):
    word_freq = {}
    for w in all_words:
        if w in words:
            word_freq[w] = words.count(w)
        else:
            word_freq[w] = 0
    return word_freq


# Computes the tf values for all words across all documents.
# A vector is generated for each document. And the tf value is for each document
def compute_tf(documents, all_words):
    tf_matrix = []

    for vec in documents:
        # tf values are calculated for each document
        content_words_count = len(vec)
        content_words_freq = compute_word_freq(vec, all_words)
        content_word_tf = dict((w, f / float(content_words_count)) for w, f in content_words_freq.items())
        tf_matrix.append(content_word_tf)

    return tf_matrix


def compute_average_freq(l_freq_1, l_freq_2):
    average_freq = {}

    keys = set(l_freq_1.keys()) | set(l_freq_2.keys())

    for k in keys:
        s_1 = l_freq_1.get(k, 0)
        s_2 = l_freq_2.get(k, 0)

        average_freq[k] = (s_1 + s_2) / 2.

    return average_freq


def js_divergence(sys_summary, doc_freq):
    summary_freq = compute_tf(sys_summary)
    average_freq = compute_average_freq(summary_freq, doc_freq)

    jsd = kl_divergence(summary_freq, average_freq) + kl_divergence(doc_freq, average_freq)
    return jsd / 2.


def n_containing(sentences, word):
    count = 0
    for sentence in sentences:
        if word in sentence:
            count = count + 1
    return count


def idf(sentences, word):
    return math.log(float(len(sentences)) / (1.0 + n_containing(sentences, word)))


def compute_tf_idf(docs):
    documents = []
    all_words = set()  # So that only unique words are stored here

    # This loop is run to get a list of all words across all documents and to represent the documents as a list of
    # individual documents containing stemmed and tokenized sentences.
    # The structure of documents is [[stemmed and tokenized doc1], [stemmed and tokenized doc2], ....]
    # The structure of all_words is [word1, word2, word3, ......]. all_words contains only unique words
    for title, doc in docs:
        sentences = []
        sentences.append(title)
        sentences.extend(doc)
        doc_stemmed = get_all_content_words_stemmed(sentences, 1)
        for word in doc_stemmed:
            all_words.add(word)
        documents.append(doc_stemmed)

    all_words = list(all_words)  # Set converted into a list. Set was used to ensure that it contains only unique words

    tf_matrix = compute_tf(documents, all_words)
    tf_idf_matrix = []
    idf_values = {}
    sentences = []

    for title, doc in docs:
        sentences.append(title)
        sentences.extend(doc)

    for word in all_words:
        idf_values[word] = idf(sentences, word)

    for tf_vec in tf_matrix:
        tf_idf_vec = dict()
        for word in tf_vec:
            tf_idf_vec[word] = tf_vec[word] * idf_values[word]
        tf_idf_matrix.append(tf_idf_vec)

    return tf_idf_matrix

def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

def fitness_fun(sys_summary, doc_freq):
    # Get tf-tdf values for the system summary
    all_words = doc_freq[0].keys()

    documents = []

    doc_stemmed = get_all_content_words_stemmed(sys_summary, 1)
    documents.append(doc_stemmed)

    tf_matrix = compute_tf(documents, all_words)
    tf_idf_matrix = []
    idf_values = {}


    for word in all_words:
        idf_values[word] = idf(sys_summary, word)

    for tf_vec in tf_matrix:
        tf_idf_vec = dict()
        for word in tf_vec:
            tf_idf_vec[word] = tf_vec[word] * idf_values[word]
        tf_idf_matrix.append(tf_idf_vec)

    sys_freq = tf_idf_matrix[0]

    similarity = 0
    for vec in doc_freq:
        similarity = similarity + cosine(sys_freq.values(), np.transpose(vec.values()))

    return similarity

