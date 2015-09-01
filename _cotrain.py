import cPickle as pickle
import numpy as np
import copy
import itertools

from collections import defaultdict
from numpy.random import RandomState

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import clone

from joblib import (delayed, Parallel)

from util import show_most_informative_features

POS_LABEL_VALUE = 1
NEG_LABEL_VALUE = 0

class CombinedClass(object):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def predict_proba(self, X1, X2):
        """X1, X2: matrix
        the probability of being positive
        """
        # print X1
        # print self.c1.predict_proba(X1)[:, 1]
        # print self.c2.predict_proba(X2)[:, 1]
        return self.c1.predict_proba(X1)[:, 1] * self.c2.predict_proba(X2)[:, 1]

    def predict(self, X1, X2):
        """return: vector"""
        return self.predict_proba(X1, X2) > 0.5


def get_confident_example_indices(clf, X, pos_n, neg_n):
    """get the confident examples and remove them from the unlabeled set
    """
    probas = clf.predict_proba(X)[:,1]#the proba for being positive
    sorted_inds = np.argsort(probas)
    pos_inds = sorted_inds[::-1][:pos_n]
    neg_inds = sorted_inds[:neg_n]
    # print probas[pos_inds]
    # print probas[neg_inds]
    assert len(pos_inds.tolist())
    assert len(neg_inds.tolist())
    return pos_inds.tolist(), neg_inds.tolist()

def train_models(model1, model2, train_X1, train_X2, y):
    trained_model1 = clone(model1)
    trained_model2 = clone(model2)
    trained_model1.fit(train_X1, y)
    trained_model2.fit(train_X2, y)
    return trained_model1, trained_model2


def bootstrap_labeled_set(model1, model2,
                          views_obj,
                          labeled_examples, unlabeled_examples,
                          pos_n, neg_n,
                          verbose=0):
    """
    Label the unlabeled set and put the top-k confident ones into the labeled set
    
    Parameters:
    ----------------
    `model1`: the first classifier

    `model2`: the first classifier

    `views_obj`: Views object

    `labeled_examples`: (views1, views2, labels) to be augmented after this function call

    `unlabeled_examples`: (views1, views2) size to be shrinked after this function call
    
    `pos_n`: number of positive examples to add

    `neg_n`: number of negative examples to add
    """
    assert len(labeled_examples) == 3
    assert len(unlabeled_examples) == 2

    view1_X, view2_X = views_obj.transform(*unlabeled_examples)

    pos_inds1, neg_inds1 = get_confident_example_indices(
        model1,
        view1_X,
        pos_n, neg_n)

    pos_inds2, neg_inds2 = get_confident_example_indices(
        model2,
        view2_X,
        pos_n, neg_n)

    pos_inds = set(pos_inds1 + pos_inds2)
    neg_inds = set(neg_inds1 + neg_inds2)
    if len(pos_inds.intersection(neg_inds)) > 0:
        print "Warning: there are overlapping between positive examples and negative examples"

    if verbose >= 1:
        print "Adding %d positive examples" % (len(pos_inds))
        for ind in pos_inds:
            print unlabeled_examples[0][ind], unlabeled_examples[1][ind]
        print ""
        print "Adding %d negative examples" % (len(neg_inds))
        for ind in neg_inds:
            print unlabeled_examples[0][ind], unlabeled_examples[1][ind]
        print ""

    def add_labeled_examples(inds, label_value):
        for ind in inds:
            labeled_examples[0].append(unlabeled_examples[0][ind])
            labeled_examples[1].append(unlabeled_examples[1][ind])
            labeled_examples[2].append(label_value)

    add_labeled_examples(pos_inds, POS_LABEL_VALUE)
    add_labeled_examples(neg_inds, NEG_LABEL_VALUE)

    # filter out those confidently-labeled items
    confident_item_inds = pos_inds.union(neg_inds)

    shrinked_unlabeled_examples = [[], []]
    for i, (v1, v2) in enumerate(zip(*unlabeled_examples)):
        if i not in confident_item_inds:
            shrinked_unlabeled_examples[0].append(v1)
            shrinked_unlabeled_examples[1].append(v2)

    unlabeled_examples = shrinked_unlabeled_examples
    # unlabeled_examples = [item
    #                       for i, item in enumerate(unlabeled_examples)
    #                       if i not in confident_item_inds]

    if verbose >= 1:
        print "Number of labeled examples: %d" % (len(labeled_examples[0]))
        print "Number of unlabeled examples: %d" % (len(unlabeled_examples[0]))

    return labeled_examples, unlabeled_examples

# This might be useful:
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
def train(clf1,
          clf2,
          views_obj,
          labeled_examples,
          unlabeled_examples,
          test_examples,
          n_iter,
          pos_n_each_iter=1, neg_n_each_iter=1,
          verbose=0, report_iter_freq=5):
    """
    Run the cotraining algorithms n_iter times
    
    Parameters:
    -------------------

    `clf1`: the first classifier

    `clf2`: the second classifier

    `views_obj`: the Views object, which transforms the features into vectors

    `labeled_examples`: list of (view1_feature, view2_feature, label)

    `unlabeled_examples`: list of (view1_feature, view2_feature)

    `test_examples`: list of (view1_feature, view2_feature, label)

    `n_iter`: maximum number of iterations

    `pos_n_each_iter`: number of positive examples to add at each iteration

    `neg_n_each_iter`: number of negative examples to add at each iteration
    
    `verbose`: debug information verbosity

    `report_iter_freq`: on every how many iterations to report

    Return:
    ------------------
    None
    
    """
    assert isinstance(labeled_examples, list)
    assert isinstance(unlabeled_examples, list)

    accuracies = []
    confusion_matrices = []

    def element2list(data):
        for i in xrange(len(data)):
            data[i] = list(data[i])
        return data

    labeled_examples = zip(*copy.deepcopy(labeled_examples))
    labeled_examples = element2list(labeled_examples)
    

    unlabeled_examples = zip(*copy.deepcopy(unlabeled_examples))
    unlabeled_examples = element2list(unlabeled_examples)
    
    # import pdb
    # pdb.set_trace()
    test_features1, test_features2, test_y = zip(*test_examples)

    i = 0
    while i<n_iter:
        view1_features, view2_features, labels = labeled_examples
        train_X1, train_X2 = views_obj.fit_transform(view1_features,
                                                     view2_features)

        # train the models separately
        trained_model1, trained_model2 = train_models(clf1, clf2, train_X1, train_X2, labels)

        if verbose >= 2:
            ## TODO:
            # how to deal with multiple views?
            print "#"*20
            print "Most informative features for clf 1:"
            show_most_informative_features(views_obj.views[0].count_vec1, trained_model1, n=10)

            print "-"*20
            print "Most informative features for clf 2:"
            show_most_informative_features(views_obj.views[0].count_vec2, trained_model2, n=10)


        # report performance
        combined_model = CombinedClass(trained_model1, trained_model2)

        test_X1, test_X2 = views_obj.transform(test_features1, test_features2)
        print test_X1.shape, test_X2.shape
        
        pred_y = combined_model.predict(test_X1, test_X2)
        accuracy = accuracy_score(test_y, pred_y)
        conf_mat = confusion_matrix(test_y, pred_y)
        
        #################
        ## Playground ###
        #################
        
        pred_y = trained_model1.predict(test_X1)
        accuracy = accuracy_score(test_y, pred_y)
        
        # record the metrics
        accuracies.append((i, accuracy))
        confusion_matrices.append(conf_mat)
        
        if i % report_iter_freq == 0:
            print "At iteration %d:" % i
            print "-" * 20
            print "Accuracy %.2f" % (accuracy*100)
            print "Confusion matrix\n: %r" % conf_mat
        
        labeled_examples, unlabeled_examples = bootstrap_labeled_set(trained_model1, trained_model2,
                                                                     views_obj,
                                                                     labeled_examples, unlabeled_examples,
                                                                     pos_n_each_iter, neg_n_each_iter,
                                                                     verbose=verbose)
        i+=1

    return clf1, clf2, accuracies, confusion_matrices

# rng = RandomState(random_state)

# views_obj, labeled_feature_items_v1, labeled_feature_items_v2, labels = pickle.load(open(labeled_path))
# print "Using views:", views_obj.view_names

# all_pos_examples = [(v1, v2, l) for v1, v2, l in zip(labeled_feature_items_v1,
#                                                      labeled_feature_items_v2,
#                                                      labels)
#                     if l == POS_LABEL_VALUE]

# all_neg_examples = [(v1, v2, l) for v1, v2, l in zip(labeled_feature_items_v1,
#                                                      labeled_feature_items_v2,
#                                                      labels)
#                     if l == NEG_LABEL_VALUE]

# pos_seed_items, pos_test_items = sample_and_split(rng, all_pos_examples,
#                                                   init_pos_n)
# neg_seed_items, neg_test_items = sample_and_split(rng, all_neg_examples,
#                                                   init_neg_n)
# seed_items = pos_seed_items + neg_seed_items
# test_items = pos_test_items + neg_test_items

# unlabeled_items = zip(*pickle.load(open(unlabeled_path)))

def run(seed_items,
        test_items,
        unlabeled_items,
        views_obj,
        n_iter,
        pos_n_each_iter, neg_n_each_iter, 
        output_path,
        verbose=0):
    """
    Params:
    -------------
    seed_items: list of (view1_feature, view2_feature, label)
    
    test_items: list of (view1_feature, view2_feature, label)

    unlabeled_items: list of (view1_feature, view2_feature)
    
    ...
    
    output_path: where the result is saved

    """
    clfs = [
        # MultinomialNB(),
        LogisticRegression(),
        # SVC(kernel='linear')
    ]

    print "started training..."
    all_results = Parallel(n_jobs=len(clfs))(delayed(train) \
                                             (clf1=clf,
                                              clf2=clf,
                                              views_obj=views_obj,
                                              labeled_examples=seed_items,
                                              unlabeled_examples=unlabeled_items,
                                              test_examples=test_items,
                                              n_iter=n_iter,
                                              pos_n_each_iter=pos_n_each_iter, 
                                              neg_n_each_iter=neg_n_each_iter,
                                              verbose=verbose,
                                              report_iter_freq=5)
                                             for clf in clfs)
    print all_results

    # pickle.dump(all_results, open('results/nb_logreg_forrest_bow_depbow.pkl', 'w'))
    pickle.dump(all_results, open(output_path, 'w'))
    
    # model, accuracies, conf_mats    

    # print "Accuracies: ", ", ".join(["%.4f" %a for _,a in accuracies])
    # print "true positive:", [m[1,1] for m in conf_mats]
    # print "true negative:", [m[0,0] for m in conf_mats]
    # print "false positive:", [m[0,1] for m in conf_mats]
    # print "false negative:", [m[1,0] for m in conf_mats]
