
def sample_from_list_and_split(items, n, rng):
    """Sample n examples from items using random generator rng
    
    >>> from numpy.random import RandomState
    >>> rng = RandomState(1) 
    >>> items = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> sample_from_list_and_split(items, 2, rng)
    (['b', 'c'], ['a', 'd', 'e', 'f'])
    """
    indices = rng.permutation(len(items))[:n]
    indices = set(indices.tolist())
    sampled = [r for i, r in enumerate(items)
               if i in indices]
    remained = [r for i, r in enumerate(items)
                if i not in indices]
    return sampled, remained


def show_most_informative_features(vectorizer, clf, n=20, sep="|"):
    """
    Copied from:
    http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
    """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    print "{sep}{:5}{sep}{:15}{sep}{:5}{sep}{:15}{sep}".format(
        'weight', 'neg term', 'weight', 'pos term', sep=sep)
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "{sep}{:.4f}{sep}{:15}{sep}{:.4f}{sep}{:15}{sep}".format(
            coef_1, fn_1, coef_2, fn_2, sep=sep)


def remove_certain_tokens(tokens, certain_tokens):
    """
    >>> remove_certain_tokens(['a', 'b', 'c'], set(['d', 'b']))
    ['a', 'c']
    """
    return [t for t in tokens if t not in certain_tokens]
