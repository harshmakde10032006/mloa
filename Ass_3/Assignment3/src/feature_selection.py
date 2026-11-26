from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=10):

    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))

    X_selected = selector.fit_transform(X, y)

    return selector, X_selected