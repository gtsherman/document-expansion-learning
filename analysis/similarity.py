import math


def make_vocab(*vectors):
    vocab = set()
    for vector in vectors:
        vocab = vocab.union(set(vector.keys()))
    return vocab


def cosine_similarity(vector1, vector2):
    vocab = make_vocab(vector1, vector2)
    numerator = sum([vector1[term] * vector2[term] for term in vocab])
    vec1_norm = sum([math.sqrt(math.pow(vector1[term], 2)) for term in vocab])
    vec2_norm = sum([math.sqrt(math.pow(vector2[term], 2)) for term in vocab])
    return numerator / (vec1_norm * vec2_norm)


def kl_divergence(true_vec, other_vec):
    """
    Assumes the vectors contain term probabilities.
    :param true_vec: A {term: prob} vector.
    :param other_vec: A different {term: prob} vector.
    :return: The KL divergence.
    """
    vocab = make_vocab(true_vec, other_vec)
    return sum([true_vec[term] * math.log2(true_vec[term] / other_vec[term]) for term in vocab])
