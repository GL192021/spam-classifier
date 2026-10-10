# Spam Classifier using Naive Bayes

This project explores spam classification using **Bernoulli and Multinomial Naive Bayes**, with three main areas of focus:

1. **Model implementation**
   - Standard implementations using scikit-learn.
   - From-scratch implementations using NumPy.
   - Alternative implementations using PyTorch.

2. **Text vectorization**
   - Binary and count-based Bag-of-Words representations.
   - TF-IDF vectorization.
   - Exploration of custom text vectorizers using NumPy and Pandas.

3. **Mathematical foundations**
   - Derivation of the probabilistic models and their parameter estimators.
   - Examination of the assumptions underlying Bernoulli and Multinomial Naive Bayes.
   - Investigation of the mathematical and probabilistic implications of different feature representations, particularly the use of TF-IDF with Multinomial Naive Bayes.

The project combines mathematical analysis, implementation, and experimental comparisons to investigate how modeling assumptions and feature representations influence classification performance.

## Closed-Form MLE Estimators
Unlike many machine-learning models, Naive Bayes does *not* require iterative optimization during training. For the models considered here, the *maximum-likelihood estimators* of the parameters can be derived in closed form. Thus, there is no gradient descent, learning rate, or sequence of training epochs: once the form of the optimal estimators is known, fitting the model amounts to computing the corresponding statistics from the training data.

We include a short mathematical note deriving the maximum-likelihood estimators for Bernoulli and Multinomial Naive Bayes. The emphasis is on the underlying probabilistic modelling assumptions and on how they lead to the closed-form formulas used by the models. See `Naive_Bayes-MLE.pdf`.

