import numpy

from nlp_utils import *

import math
import numpy as np


def stem(sentences, N=1):
    def is_ngram_content(g):
        for a in g:
            if not (a in stopset):
                return True
        return False

    all_words = []
    all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(sentences)])

    if N == 1:
        content_words = [w for w in all_words if w not in stopset]
    else:
        content_words = all_words

    normalized_content_words = map(normalize_word, content_words)
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N) if is_ngram_content(gram)]
    return normalized_content_words


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


def n_containing(sentences, word):
    count = 0
    for sentence in sentences:
        if word in sentence:
            count = count + 1
    return count


def idf(sentences, word):
    return math.log(float(len(sentences)) / (1.0 + n_containing(sentences, word)))


def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))


def fitness(sys_summary, docs_weight_matrix, docs, N, M):
    sys_tf_idf_matrix, all_words = generate_sys_tf_idf(sys_summary)
    titles_tf_idf_matrix = generate_titles_tf_idf(sys_summary, docs, all_words)

    sys_summary_weights = generate_weight_matrix(sys_tf_idf_matrix)

    S = len(sys_summary)

    trf = compute_trf(sys_tf_idf_matrix, titles_tf_idf_matrix)
    cf = compute_cf(sys_summary_weights, M)
    rf = compute_rf(sys_tf_idf_matrix, docs_weight_matrix, N, S)

    return fitness_fun(1.25, 0.80, 1, trf, cf, rf)


def fitness_fun(alpha, beta, gamma, trf, cf, rf):
    F = ((alpha * trf) + (beta * cf) + (gamma * rf)) / (alpha + beta + gamma)
    return F


def generate_doc_tf_idf(docs):
    doc_documents = []
    all_documents = []
    doc_all_words = set()  # So that only unique words are stored here

    # The structure of documents is [[stemmed and tokenized doc1], [stemmed and tokenized doc2], ....]
    # The structure of all_words is [word1, word2, word3, ......]. all_words contains only unique words
    for title, doc in docs:
        for doc_sentence in doc:
            all_documents.append(doc_sentence)
            doc_stemmed = stem(doc_sentence, 1)
            for word in doc_stemmed:
                doc_all_words.add(word)
            doc_documents.append(doc_stemmed)

    doc_all_words = list(
        doc_all_words)  # Set converted into a list. Set was used to ensure that it contains only unique words

    doc_tf_matrix = compute_tf(doc_documents, doc_all_words)
    doc_tf_idf_matrix = []
    doc_idf_values = {}

    for word in doc_all_words:
        doc_idf_values[word] = idf(all_documents, word)

    for doc_tf_vec in doc_tf_matrix:
        doc_tf_idf_vec = dict()
        for doc_word in doc_tf_vec:
            doc_tf_idf_vec[doc_word] = doc_tf_vec[doc_word] * doc_idf_values[doc_word]
        doc_tf_idf_matrix.append(doc_tf_idf_vec)

    return doc_tf_idf_matrix


def generate_sys_tf_idf(sys_summary):
    # Get tf-idf values for the system summary
    documents = []
    all_words = set()  # So that only unique words are stored here

    # The structure of documents is [[stemmed and tokenized doc1], [stemmed and tokenized doc2], ....]
    # The structure of all_words is [word1, word2, word3, ......]. all_words contains only unique words
    for sentence in sys_summary:
        doc_stemmed = stem(sentence, 1)
        for word in doc_stemmed:
            all_words.add(word)
        documents.append(doc_stemmed)

    all_words = list(all_words)  # Set converted into a list. Set was used to ensure that it contains only unique words

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

    return tf_idf_matrix, all_words


def generate_titles_tf_idf(sys_summary, docs, all_words):
    titles = []
    # Get all titles
    for doc in docs:
        doc_title = doc[0]
        titles.append(doc_title)

    titles_stemmed = []
    for title in titles:
        title_stemmed = stem(title, 1)
        titles_stemmed.append(title_stemmed)

    titles_tf_matrix = compute_tf(titles_stemmed, all_words)
    titles_tf_idf_matrix = []
    titles_idf_values = {}

    for word in all_words:
        titles_idf_values[word] = idf(sys_summary, word)

    for titles_tf_vec in titles_tf_matrix:
        titles_tf_idf_vec = dict()
        for word in titles_tf_vec:
            titles_tf_idf_vec[word] = titles_tf_vec[word] * titles_idf_values[word]
        titles_tf_idf_matrix.append(titles_tf_idf_vec)

    return titles_tf_idf_matrix


def generate_weight_matrix(sys_summary_tf_idf_values):
    weights = []
    sys_summary_len = len(sys_summary_tf_idf_values)
    for i in range(0, sys_summary_len - 1):
        for j in range(i, sys_summary_len - 1):
            cos_sim = cosine(sys_summary_tf_idf_values[j].values(),
                             np.transpose(sys_summary_tf_idf_values[j + 1].values()))
            weights.append(cos_sim)

    return weights


# Compute Topic relation Factor(TRF)
def compute_trf(sys_summary_tf_idf_values, titles_tf_idf_matrix):
    S = len(sys_summary_tf_idf_values)
    n = len(titles_tf_idf_matrix)
    trf = 0

    for q in titles_tf_idf_matrix:
        tr = 0
        for Sj in sys_summary_tf_idf_values:
            tr = tr + cosine(Sj.values(), np.transpose(q.values()))
        trf = trf + tr / S

    trf = trf / n

    return trf


# Compute Cohesion Factor (CF)
def compute_cf(sys_summary_weights, M):
    Cs = 0
    Ns = len(sys_summary_weights)
    for W in sys_summary_weights:
        Cs = Cs + W

    Cs = Cs / Ns

    CFs = math.log((Cs * 9 + 1), 10) / math.log((M * 9 + 1), 10)

    return CFs


def sim(docs_representation, i, j):
    if i >= j:
        return 0
    else:
        cosine_sim = cosine(docs_representation[i].values(), np.transpose(docs_representation[j].values()))
        return cosine_sim


def compute_longest_path_weight(docs_representation):
    m = len(docs_representation)
    n = len(docs_representation)


    A = [[0 for x in range(n)] for y in range(m)]

    for j in range(1, n):
        for i in range(1, m):
            for k in range(1, i - 1):
                if A[i][j] < A[k][j - 1] + sim(docs_representation, k, i):
                    A[i][j] = A[i][j - 1] + sim(docs_representation, k, i)

    return A


# Computer Readability Factor(RF)
def compute_rf(sys_tf_idf_matrix, docs_weight_matrix, N, S):
    Rs = 0
    sys_summary_len = len(sys_tf_idf_matrix)
    for i in range(0, sys_summary_len-1):
        cos_sim = cosine(sys_tf_idf_matrix[i].values(),
                             np.transpose(sys_tf_idf_matrix[i + 1].values()))
        Rs = Rs + cos_sim

    maxCol = []
    for i in range(N - 1):
        maxCol.append(docs_weight_matrix[i][S-1])

    Rmax = max(maxCol)
    RF = Rs / Rmax

    return RF