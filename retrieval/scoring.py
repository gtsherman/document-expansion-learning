import math


class DirichletTermScorer(object):
    def __init__(self, index, mu=2500, epsilon=1.0):
        self.index = index
        self.mu = mu
        self.epsilon = epsilon

    def score(self, term, document):
        term = term.lower()

        doc_vector = document.document_vector()
        term_freq = doc_vector[term]
        doc_length = sum(doc_vector.values())
        collection_prob = (self.epsilon + self.index.term_count(term)) / self.index.total_terms()
        return (term_freq + self.mu * collection_prob) / (doc_length + self.mu)


class InterpolatedTermScorer(object):
    def __init__(self, scorers):
        """
        :param scorers: A {scorer: weight} dict.
        """
        self.scorers = scorers

    def score(self, term, document):
        return sum([self.scorers[scorer] * scorer.score(term, document) for scorer in self.scorers])


class ExpansionDocTermScorer(object):
    def __init__(self, scorer):
        self._scorer = scorer

    def score(self, term, expansion_docs):
        return sum([exp_score * self._scorer.score(term, exp_doc) for exp_doc, exp_score in expansion_docs])


class QLQueryScorer(object):
    def __init__(self, term_scorer):
        self.term_scorer = term_scorer

    def score(self, query, document):
        score = 0.0
        for term in query.vector:
            q_weight = query.vector[term] / query.length()
            term_score = math.log(self.term_scorer.score(term, document))
            score += q_weight * term_score
        return score
