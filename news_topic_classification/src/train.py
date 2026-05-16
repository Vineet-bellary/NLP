from sklearn.linear_model import LogisticRegression
import joblib

from util.data_prep import prepare_data
from util.vectorize import build_vectorizer
from util.logger import setup_logging
import config

# Configurations
train_df_path = config.TRAIN_DATA
model_state_path = config.LOGISTIC_REGRESSION_MODEL

vectorizer = build_vectorizer()

logger = setup_logging("train")

# train data prep
x_train, y_train = prepare_data(train_df_path)
x_train_vectorized = vectorizer.fit_transform(x_train)

model = LogisticRegression(max_iter=1000, random_state=42)
logger.info(model)

# train the model
logger.info("Training the model...")
model.fit(x_train_vectorized, y_train)
logger.info("Model training completed.")

model_save = {
    "model": model,
    "vectorizer": vectorizer,
}

joblib.dump(model_save, model_state_path)
logger.info(f"Model saved to {model_state_path}")
