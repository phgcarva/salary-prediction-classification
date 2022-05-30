from sklearn.base import TransformerMixin, BaseEstimator


class NumCatEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols, default_missing=-1):
        self.cols = cols
        self.vocab = dict([(col, {}) for col in self.cols])
        self.default_missing = default_missing

    def fit(self, X, y=None):
        for col in self.cols:
            self.vocab[col].update(
                dict([(val, i) for i, val in enumerate(X[col].unique())])
            )

        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.cols:
            X_[col] = X_[col].apply(
                lambda x: self.vocab[col][x]
                if x in self.vocab[col]
                else self.default_missing
            )
        return X_
