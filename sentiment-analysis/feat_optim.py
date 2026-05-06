from pathlib import Path
import string

from util.eval import feature_importance
from util.preprocessing import preprocess_pipeline
from util.vectorizer import vectorize_texts

DIR_DATA = Path("data").resolve()
MODEL_DIR = Path("model").resolve()
DATA_FILE = DIR_DATA / "IMDB_dataset.csv"

sentiment_analysis = preprocess_pipeline(DATA_FILE)

if sentiment_analysis is not None and sentiment_analysis.empty is False:
    print("Data loaded successfully....")
    print(sentiment_analysis.head(10))
else:
    print("Data loading failed. Exiting.")
    exit(1)

vectorizer, x_train, x_test, y_train, y_test = vectorize_texts(sentiment_analysis)

feature_importance(vectorizer, 10)

# punctuation = string.punctuation
# print("Punctuation characters:", punctuation)
