import argparse

import gensim
import pyndri


class IndriDocuments(object):
    def __init__(self, index):
        self.index = index
        self.dictionary = pyndri.extract_dictionary(index)

    def __iter__(self):
        """
        Iterate over every document in the collection, turning it into a TaggedDocument object. The terms of the
        document are used as words, and the document's internal ID is used as the tag since it meets the criteria of
        a unique integer identifier, as requested by gensim.

        This works by treating the generator as an iterator. That is, when iter() is called on IndriDocuments,
        it "returns" a generator. When next() is called on the generator, it works as a generator always does,
        by yielding the next item (document). But this works better for gensim, which needs to iterate over the items
        multiple times, because the next time that iter() is called on IndriDocuments, a new generator is created
        that will start over from the first document.
        """
        for doc_id in range(self.index.document_base(), self.index.maximum_document()):
            _, term_ids = self.index.document(doc_id)
            yield gensim.models.doc2vec.TaggedDocument(words=[self.dictionary[term_id] for term_id in term_ids if
                                                              term_id > 0], tags=[doc_id])


def main():
    options = argparse.ArgumentParser(description='Generate doc2vec word embeddings using an Indri index.')
    required = options.add_argument_group('required arguments')
    required.add_argument('-i', '--index', required=True, help='The Indri index to use.')
    required.add_argument('-n', '--model-name', required=True, help='The name to use when saving the model.')
    options.add_argument('-w', '--workers', required=False, type=int, default=3, help='The number of workers to use.')
    args = options.parse_args()

    index = pyndri.Index(args.index)

    doc2vec = gensim.models.Doc2Vec(IndriDocuments(index), workers=args.workers, vector_size=args.s)
    doc2vec.save(args.model_name)


if __name__ == '__main__':
    main()
