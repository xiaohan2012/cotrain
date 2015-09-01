# coding: utf-8

import nltk
import codecs
from numpy.random import RandomState

from view_extraction.views import Views
from view_extraction.case import CapitalizedBoWView
from view_extraction.data_record import DataRecord
from util import remove_certain_tokens

from _cotrain import run


stopwords = set(nltk.corpus.stopwords.words()+['fraud'])
stemmer = nltk.stem.porter.PorterStemmer()


def process_sentence(sent):
    """Pre-process the sentence
    """
    tokens = nltk.word_tokenize(sent)

    # stemming
    tokens = map(stemmer.stem, tokens)
    
    # remove stopwords
    # important
    tokens = remove_certain_tokens(tokens, stopwords)
    
    return tokens
    

def load_labeled_data(readable_obj):
    u"""
    >>> from StringIO import StringIO
    >>> readable = StringIO(u"1---\\tI love you\\n0---\\tYou love you, too. €")
    >>> drs = load_labeled_data(readable)
    >>> len(drs)
    2
    >>> drs[0].tokens
    [u'I', u'love', u'you']
    >>> drs[0].y
    1
    """
    drs = []
    for l in readable_obj:
        if l.strip():
            try:
                y_str, sent = l.split('\t')
                y = int(y_str.strip('---'))
            except ValueError:
                print l
                raise
            tokens = process_sentence(sent)
            drs.append(DataRecord(tokens, y))
    return drs
 

def load_unlabeled_data(readable_obj):
    u"""
    >>> from StringIO import StringIO
    >>> readable = StringIO(u"I love you\\nYou love you, too. €")
    >>> drs = load_unlabeled_data(readable)
    >>> len(drs)
    2
    >>> drs[1].tokens
    [u'You', u'love', u'you', u',', u'too', u'.', u'\\u20ac']
    >>> drs[1].y
    """
    drs = []
    for l in readable_obj:
        tokens = process_sentence(l)
        drs.append(DataRecord(tokens))
    return drs


def load_data(labeled_path, unlabeled_path):
    """
    
    Return:
    -----
    (list of DataRecord, list of unlabeled DataRecord): 
        labeled and unlabeled data
    """
    return load_labeled_data(codecs.open(labeled_path, 'r', 'utf8')),\
        load_unlabeled_data(codecs.open(unlabeled_path, 'r', 'utf8'))


def main():    
    # Misc
    view_extractors = [CapitalizedBoWView()]
    views = Views(view_extractors)

    seed_ids = [1, 6,  # pos
                2, 12, 35  # neg
    ]

    # load data
    labeled_path = 'data/fraud/fraud-labeled'
    unlabeled_path = 'data/fraud/fraud-unlabeled-2000'

    ldrs, udrs = load_data(labeled_path, unlabeled_path)
    
    # Split seed and test data
    seed_drs = [ldrs[_id] for _id in seed_ids]
    test_drs = [ldrs[_id] for _id in xrange(len(ldrs)) 
                if _id not in set(seed_ids)]
    
    # Extract views
    seed_views1, seed_views2 = views.extract(seed_drs)
    test_views1, test_views2 = views.extract(test_drs)
    unlabeled_views1, unlabeled_views2 = views.extract(udrs)
    
    # Transpose them
    seed_items = zip(seed_views1, seed_views2,
                     [dr.y for dr in seed_drs])
    test_items = zip(test_views1, test_views2,
                     [dr.y for dr in test_drs])
    unlabeled_items = zip(unlabeled_views1, unlabeled_views2)

    # Run!
    run(seed_items,
        test_items,
        unlabeled_items,
        views,
        n_iter=100,
        pos_n_each_iter=1, neg_n_each_iter=1,
        output_path="result/fraud.pkl",
        verbose=2)

if __name__ == "__main__":
    main()
