

class View(object):
    name = 'view'

    def __init__(self):
        self.name = self.__class__.name

    def extract(self, data_record):
        """Extract the features for two views from one single data_record"""
        raise NotImplementedError

    def fit(self, view1_features, view2_features):
        """Learn about the view features of one/multiple data items"""
        raise NotImplementedError

    def transform(self, view1_features, view2_features):
        """Transform the view features of one/multiple data items into matrix"""
        raise NotImplementedError

    def fit_transform(self, view1_features, view2_features):
        self.fit(view1_features, view2_features)
        return self.transform(view1_features, view2_features)
