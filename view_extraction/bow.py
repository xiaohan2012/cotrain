from base import View

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class BagOfWordView(View):
    """
    View that process words(stemming, lowercasing) and count each word's frequency
    """
    def __init__(self, *args, **kwargs):
        self.count_vec1 = None
        self.count_vec2 = None

        self.tfidf_vec1 = None
        self.tfidf_vec2 = None

        super(BagOfWordView, self).__init__(*args, **kwargs)

    def fit(self, v1, v2, use_idf=False):
        """
        v1, v2: both should be string|unicode, which is required by CountVectorizer.fit
        """
        ## TODO: add `use_tf` option
        self.count_vec1 = CountVectorizer().fit(v1)
        self.count_vec2 = CountVectorizer().fit(v2)

        self.tfidf_vec1 = TfidfTransformer(use_idf=use_idf).fit(
            self.count_vec1.transform(v1))

        self.tfidf_vec2 = TfidfTransformer(use_idf=use_idf).fit(
            self.count_vec2.transform(v2))

        return self

    def transform(self, v1, v2):
        return self.tfidf_vec1.transform(self.count_vec1.transform(v1)), \
            self.tfidf_vec2.transform(self.count_vec2.transform(v2))
