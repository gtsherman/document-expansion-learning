import argparse

import gensim
import pyndri
import sys


def main():
    options = argparse.ArgumentParser(description='Get similar docs according to doc2vec cosine similarity.')
    required = options.add_argument_group('required arguments')
    required.add_argument('-i', '--index', required=True, help='The index to search for similar docs.')
    required.add_argument('-n', '--model-name', required=True, help='The pre-trained doc2vec model name. This model '
                                                                    'should be trained on the same index as --index.')
    required.add_argument('-d', '--docs', required=True, help='A file containing the document IDs to "expand."')
    options.add_argument('-k', '--top-k', required=False, type=int, default=50, help='The number of similar documents.')
    options.add_argument('-o', '--outside-index', required=False, help='The outside index to which the document IDs '
                                                                       'belong. If not given, the value of --index '
                                                                       'will be used.')
    args = options.parse_args()

    sim_doc_index = pyndri.Index(args.index)
    doc_text_index = sim_doc_index
    if args.outside_index is not None:
        doc_text_index = pyndri.Index(args.outside_index)
        dictionary = pyndri.extract_dictionary(doc_text_index)

    d2v = gensim.models.doc2vec.Doc2Vec.load(args.model_name)
    dv = d2v.docvecs
    dv.init_sims(replace=True)
    if args.outside_index is None:
        del d2v  # save memory, but can only do this if we don't need to infer a vector

    with open(args.docs) as f:
        docs = f.read().strip().split('\n')

    for doc in docs:
        try:
            doc_id = doc_text_index.document_ids((doc,))[0][1]
        except IndexError:
            print('Issue getting docID from tuple:', doc_text_index.document_ids((doc,)), '. Docno:', doc,
                  file=sys.stderr)
            continue

        lookup_doc = doc_id  # as long as we're using the same index for lookup doc and sim docs, just need the doc ID
        if args.outside_index is not None:  # if we're using an inferred document vector, we first need to infer it
            _, term_ids = doc_text_index.document(doc_id)
            inferred_vector = d2v.infer_vector([dictionary[term_id] for term_id in term_ids if term_id > 0], epochs=50)
            lookup_doc = (inferred_vector, 1.0)  # this is the format needed to lookup sim docs with inferred vector

        similar_docs = dv.most_similar(positive=[lookup_doc], topn=args.top_k)

        for similar_doc, cosine_score in similar_docs:
            try:
                similar_docno = sim_doc_index.ext_document_id(similar_doc)
                print(doc, similar_docno, cosine_score, sep=',')
            except IndexError:
                print('Issue with docID:', similar_doc, file=sys.stderr)


if __name__ == '__main__':
    main()
