import argparse

import gensim
import pyndri
import sys


def main():
    options = argparse.ArgumentParser(description='Get similar docs according to doc2vec cosine similarity.')
    required = options.add_argument_group('required arguments')
    required.add_argument('-i', '--index', required=True, help='The index to use.')
    required.add_argument('-n', '--model-name', required=True, help='The pre-trained doc2vec model name.')
    required.add_argument('-d', '--docs', required=True, help='A file containing the documents to "expand."')
    options.add_argument('-k', '--top-k', required=False, type=int, default=50, help='The number of similar documents to identify.')
    args = options.parse_args()

    index = pyndri.Index(args.index)

    d2v = gensim.models.doc2vec.Doc2Vec.load(args.model_name)
    dv = d2v.docvecs
    del d2v  # save memory

    with open(args.docs) as f:
        docs = f.read().split('\n')

    for doc in docs:
        try:
            doc_id = index.document_ids((doc,))[0][1]
        except IndexError:
            print('Issue getting docID from tuple:', index.document_ids((doc,)), '. Docno:', doc, file=sys.stderr)
            continue
        similar_docs = dv.most_similar(positive=[doc_id], topn=args.top_k)
        for similar_doc, cosine_score in similar_docs:
            try:
                similar_docno = index.ext_document_id(similar_doc)
                print(doc, similar_docno, cosine_score, sep=',')
            except IndexError:
                print('Issue with docID:', similar_doc, file=sys.stderr)


if __name__ == '__main__':
    main()
