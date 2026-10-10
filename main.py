import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from from_scratch_models import BernoulliNaiveBayes, MultinomialNaiveBayes

df = pd.read_csv(
    "data/spam.csv",
    sep="\t",
    header=None,
    names=["label", "text"]
)
### my saved csv file, only separate the label form the text by tab.
### here, with  'sep=\t' I give this exact instruction.
### so it separates the label from the text.
### then I put the column names "label", "text"




labels_dict = {
    "ham": 0,
    "spam": 1
}

# df["label"] = df["label"].map(labels_dict)
df["label"] = (df["label"] == "spam").astype(int)
# print(df.head())
# print(df.info())
# print(df['text'].value_counts())
# print(df["label"].value_counts())
# print(df["label"].isna().sum())
# print(df["label"])
# print(df["label"].values)
# print(df[df["text"]==""].value_counts())




# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=7,
    stratify=df["label"]
)



#### Convert text into TF-IDF features (for Multinomial)
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

#### text to binary vector (for Bernoulli)
count_vectorizer = CountVectorizer(binary=False, stop_words="english")

#### text to binary vector (for Bernoulli)
binary_vectorizer = CountVectorizer(binary=True, stop_words="english")



X_train_M_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_M_tfidf = tfidf_vectorizer.transform(X_test)

X_train_M_count = count_vectorizer.fit_transform(X_train)
X_test_M_count = count_vectorizer.transform(X_test)

X_train_B = binary_vectorizer.fit_transform(X_train)
X_test_B = binary_vectorizer.transform(X_test)

# print(type(X_train))
# print(type(X_train_M))
# print(type(X_train_B))

# y_tr_arr = np.array(y_train)
# # print(X_train.shape)
# print(X_train_B.shape)
# print(X_train_B[y_tr_arr==0])
# print(X_train_B[1])
# print(type(X_train_B.data))
# # print(X_train_M_tfidf)
# # print(X_train_B)


# train
model_multinomial_learnt_prior = MultinomialNB()
model_multinomial_learnt_prior.fit(X_train_M_tfidf, y_train)

model_multinomial_learnt_prior_from_scratch = MultinomialNaiveBayes()
model_multinomial_learnt_prior_from_scratch.fit(X_train_M_tfidf, y_train)

model_bernoulli = BernoulliNB()
model_bernoulli.fit(X_train_B, y_train)

model_Bernoulli_from_scratch = BernoulliNaiveBayes()
model_Bernoulli_from_scratch.fit(X_train_B, y_train)

# predict
predictions_multinomial = model_multinomial_learnt_prior.predict(X_test_M_tfidf)
predictions_multinomial_from_scratch = model_multinomial_learnt_prior_from_scratch.predict(X_test_M_tfidf)
predictions_bernoulli = model_bernoulli.predict(X_test_B)
predictions_bernoulli_from_scratch = model_Bernoulli_from_scratch.predict(X_test_B)

print(predictions_bernoulli)
print(type(predictions_bernoulli))

# evaluate
print(classification_report(y_test, predictions_bernoulli))
print()
print("="*150)
print("="*150)
print("="*150)
print()
# print(classification_report(y_test, predictions_multinomial))
print(classification_report(y_test, predictions_bernoulli_from_scratch))



# ## compare predictions of the two Bernoulli implementations: scikit and from-scratch
# # ideally prints 1.0
# print(np.mean(predictions_bernoulli == predictions_bernoulli_from_scratch))
#
# #compare parameters from each implemanations
# # ideally both True
# print(np.allclose(model_Bernoulli_from_scratch.feature_prob, np.exp(model_bernoulli.feature_log_prob_)))
# print(np.allclose(model_Bernoulli_from_scratch.class_log_prior, model_bernoulli.class_log_prior_))
#


## compare predictions of the two Multinomial implementations: scikit and from-scratch
# ideally prints 1.0
print(np.mean(predictions_multinomial == predictions_multinomial_from_scratch))

#compare parameters from each implemanations
# ideally both True
print(np.allclose(model_multinomial_learnt_prior_from_scratch.feature_prob, np.exp(model_multinomial_learnt_prior.feature_log_prob_)))
print(np.allclose(model_multinomial_learnt_prior_from_scratch.class_log_prior, model_multinomial_learnt_prior.class_log_prior_))
