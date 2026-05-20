from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from fake_news_detector.config import RANDOM_STATE


def build_logistic_regression_model() -> LogisticRegression:
    """Build and return a Logistic Regression model"""

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    return model


def build_naive_bayes_model() -> MultinomialNB:
    """Build and return a Naive Bayes model"""

    model = MultinomialNB()

    return model


def build_linear_svm_model() -> LinearSVC:
    """Build and return a Linear SVM model"""

    model = LinearSVC(random_state=RANDOM_STATE)

    return model
