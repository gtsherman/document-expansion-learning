import argparse

import gensim
import pyndri
import pyndri.compat


def main():
    options = argparse.ArgumentParser(description='Generate word2vec word embeddings using an Indri index.')
    required = options.add_argument_group('required arguments')
    required.add_argument('-i', '--index', required=True, help='The Indri index to use.')
    required.add_argument('-n', '--model-name', required=True, help='The name to use when saving the model.')
    args = options.parse_args()

    index = pyndri.Index(args.index)
    sentences = pyndri.compat.IndriSentences(index, pyndri.extract_dictionary(index))

    word2vec = gensim.models.Word2Vec(sentences, sg=1, negative=5)
    word2vec.save(args.model_name)


if __name__ == '__main__':
    main()
