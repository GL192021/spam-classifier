import numpy as np

class BernoulliNaiveBayes:
    '''
    expects already a CSR matrix of training set of
    original size n\times d
    n: number of observations
    d: length of vocabulary
    row i represents the observation X^{(i)}
    column j represents the word w_j.
    This is a compressed sparse 0-1 matrix
    '''

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, prior=None):
        y = np.asarray(y)

        ## making a sparse matrix into binarized matrix. if it is already binarized, nothing changes.
        X = X.copy()
        X.data = np.ones_like(X.data) ## the CSR matrix, changes its values to 1

        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        k_classes = len(self.classes)


        self.class_log_prior = np.zeros(k_classes)

        self.feature_prob = np.zeros((k_classes, n_features))

        for k_idx, k in enumerate(self.classes):
            X_k = X[y == k]
            N_k = X_k.shape[0]

            if prior is None:
                # pi_k = N_k / n
                self.class_log_prior[k_idx] = np.log(N_k / n_samples)
            else:
                try:
                    self.class_log_prior[k_idx] = np.log(prior[k_idx])
                except ValueError:
                    print("Classes - prior selected mismatch")
            # Number of class-k documents containing each word
            word_presence_count = X_k.sum(axis=0)

            # Bernoulli MLE with additive smoothing
            self.feature_prob[k_idx] = (
                word_presence_count + self.alpha
            ) / (
                N_k + 2 * self.alpha
            )

        return self

    # The theoretical model need the likelihood P(X|Y). In practice this is relative to P(X,Y)
    def joint_log_likelihood(self, X):
        # X = np.asarray(X)
        X = X.copy()
        X.data = np.ones_like(X.data)

        log_theta = np.log(self.feature_prob)
        log_one_minus_theta = np.log(1 - self.feature_prob)

        # return (
        #     X @ log_theta.T
        #     + (1 - X) @ log_one_minus_theta.T
        #     + self.class_log_prior
        # )
        ## 1-X works with arrays, but not with CSR matrices
        ## not merely a sparse-matrix trick; it is a direct and faithful algebraic reformulation of the Bernoulli log-likelihood
        return (
                X @ (log_theta - log_one_minus_theta).T
                + log_one_minus_theta.sum(axis=1)
                + self.class_log_prior
        )

    def predict(self, X):
        log_scores = self.joint_log_likelihood(X)
        best_class_idx = np.argmax(log_scores, axis=1)

        return self.classes[best_class_idx]



class MultinomialNaiveBayes:
    '''
    expects already a CSR matrix of training set of
    original size n\times d
    n: number of observations
    d: length of vocabulary
    row i represents the observation X^{(i)}
    column j represents the word w_j.
    This is a compressed sparse 0-1 matrix
    '''

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, prior=None):
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        d_classes = len(self.classes)

        self.class_log_prior = np.zeros(d_classes)

        self.feature_prob = np.zeros((d_classes, n_features))

        for k_idx, k in enumerate(self.classes):
            X_k = X[y == k]
            N_k = X_k.shape[0]
            tot_tokens_N_k = (X_k.sum(axis=0)).sum()

            if prior is None:
                # pi_k = N_k / n
                self.class_log_prior[k_idx] = np.log(N_k / n_samples)
            else:
                try:
                    self.class_log_prior[k_idx] = np.log(prior[k_idx])
                except ValueError:
                    print("Classes - prior selected mismatch")
            # Number of class-k documents containing each word
            word_count = X_k.sum(axis=0)

            # Multinomial MLE with additive smoothing
            self.feature_prob[k_idx] = (
                word_count + self.alpha
            ) / (
                tot_tokens_N_k + self.alpha * n_features
            )

        return self

    # The theoretical model need the likelihood P(X|Y). In practice this is relative to P(X,Y)
    def joint_log_likelihood(self, X):

        log_theta = np.log(self.feature_prob)

        return (
                X @ log_theta.T + self.class_log_prior
        )

    def predict(self, X):
        log_scores = self.joint_log_likelihood(X)
        best_class_idx = np.argmax(log_scores, axis=1)

        return self.classes[best_class_idx]
