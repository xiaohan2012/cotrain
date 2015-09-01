import cPickle as pkl
import numpy as np
from collections import (Counter, defaultdict)
from scipy.sparse import (hstack, issparse, csr_matrix, csc_matrix)
from sklearn.feature_extraction import DictVectorizer


from mynlp.dependency.tree import NodeNotFoundError
from mynlp.string_util.multistring_matching import MultiStringMatcher

from data import DataRecord


class ViewExtracionError(Exception):
    pass





class FeatureItem(dict):
    """
    One data item that contains features of different types
    """

class Views(object):
    """

    >>> record_paths = ['test/data/normal.txt.out',
    ... 'test/data/dependency_bow_view_extraction.txt.out']
    >>> paths1 = [('nsubj',), ('nsubj', 'nn'), ('vmod', ), ('vmod', 'dobj'), ('vmod', 'dobj', 'nn')]
    >>> paths2 = [('dobj', ), ('dobj', 'nn'), ('dobj', 'amod'), ('prep_of',), ('prep_for', ), ('prep_for', 'conj_and')]
    >>> dep_view = DependencyBoWView(paths1, paths2)
    >>> bow_view = ProdClfBowView()
    >>> objsect_view = ObjectDistributionView('product_classification/feature_extractors/pickle/product_vect.pkl', 'obj2sect')
    >>> compsect_view = CompanyDistributionView('product_classification/feature_extractors/pickle/company_vect.pkl', 'comp2sec')
    >>> views = Views((dep_view, bow_view, objsect_view, compsect_view))

    >>> data_records = [DataRecord.from_parse_tree_path(path) for path in record_paths]
    >>> features_items1, features_items2 = views.extract(data_records)
    >>> views.fit_transform(features_items1, features_items2) # doctest: +ELLIPSIS
    (<2x541 sparse matrix of type '<type 'numpy.float64'>'...
    """
    def __init__(self, views):
        self.views = views
        assert len(self.view_names) == len(set(self.view_names)), "duplicate view names in %r" % self.view_names

    def extract(self, data_records):
        """
        Parameter:
        -------------
        data_records: list of DataRecord

        Return:
        ------------
        (list of FeatureItem, list of FeatureItem): data on the two views
        """
        feature_items1 = []
        feature_items2 = []
        for data_record in data_records:
            item_view1 = FeatureItem()
            item_view2 = FeatureItem()
            for view in self.views:
                view1_data, view2_data = view.extract(data_record)
                item_view1[view.name] = view1_data
                item_view2[view.name] = view2_data

            feature_items1.append(item_view1)
            feature_items2.append(item_view2)

        return feature_items1, feature_items2

    def _extract_feature_values(self, feature_name, items):
        return [item[feature_name] for item in items]

    def fit(self, feature_items1, feature_items2):
        """
        feature_items1, feature_items2: list of FeatureItem

        Return:
        self
        """
        for view in self.views:
            view.fit(self._extract_feature_values(view.name, feature_items1),
                     self._extract_feature_values(view.name, feature_items2))
        return self

    def transform(self, feature_items1, feature_items2):
        """
        feature_items1, feature_items2: list of FeatureItem

        Return:
        (matrix, matrix)
        """
        view1_matrices = []
        view2_matrices = []
        for view in self.views:
            m1, m2 = view.transform(
                self._extract_feature_values(view.name, feature_items1),
                self._extract_feature_values(view.name, feature_items2))
            if m1 is not None:
                view1_matrices.append(m1)
            if m2 is not None:
                view2_matrices.append(m2)

        view1_matrix = hstack(tuple(view1_matrices))
        view2_matrix = hstack(tuple(view2_matrices))
        return (view1_matrix, view2_matrix)

    def fit_transform(self, feature_items1, feature_items2):
        """
        feature_items1, feature_items2: list of FeatureItem
        """
        self.fit(feature_items1, feature_items2)
        return self.transform(feature_items1, feature_items2)

    def extract_fit_transform(self, data_records):
        v1, v2 = self.extract(data_records)
        return self.fit_transform(v1, v2)

    @property
    def view_names(self):
        return [v.name for v in self.views]
