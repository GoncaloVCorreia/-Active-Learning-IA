from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError
import rubrix as rb
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("data/PSY.csv")
test_df = pd.read_csv("data/PSY.csv")
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
# Define our classification model
#classifier = MultinomialNB()
classifier=MultinomialNB()

# Define active learner
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=entropy_sampling
)
# The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)
vectorizer = CountVectorizer(ngram_range=(1, 5))# The resulting matrices will have the shape of (`nr of examples`, `nr of word n-grams`)

X_train = vectorizer.fit_transform(train_df.CONTENT)
X_test = vectorizer.transform(test_df.CONTENT)

# Number of instances we want to annotate per iteration
n_instances = 10

# Accuracies after each iteration to keep track of our improvement
accuracies = []
x=0

query_idx, query_inst = learner.query(X_train, n_instances=n_instances)

# get predictions for the queried examples
try:
    probabilities = learner.predict_proba(X_train[query_idx])
# For the very first query we do not have any predictions
except NotFittedError:
    probabilities = [[0.5, 0.5]]*n_instances
        

records = [
    rb.TextClassificationRecord(
        id=idx,
        text=train_df.CONTENT.iloc[idx],
        prediction=list(zip(["HAM", "SPAM"], probs)),
        prediction_agent="MultinomialNB",
    )
    for idx, probs in zip(query_idx, probabilities)
]

# Log the records
rb.log(records, name="active_learning_tutorial")


# Load the annotated records into a pandas DataFrame
records_df = rb.load("active_learning_tutorial", ids=query_idx.tolist()).to_pandas()


