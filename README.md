# NLP Learning Projects

This repository is my NLP learning workspace containing multiple small-to-medium projects.
Each folder is a standalone experiment or mini application focused on a specific NLP task.

## Projects

- `Fake_News_Detection/`: Fake news detection pipeline with model comparison, training, prediction, and error analysis.
- `news_topic_classification/`: Topic classification model for news articles.
- `sentiment-analysis/`: Sentiment analysis on IMDB-style reviews.
- `spam-classifier/`: SMS/email spam classification project.
- `text-cleaner/`: Text cleaning utility scripts and demo input.

## Repository Structure

```text
NLP/
|-- README.md
|-- .gitignore
|-- Fake_News_Detection/
|-- news_topic_classification/
|-- sentiment-analysis/
|-- spam-classifier/
`-- text-cleaner/
```

## Common Workflow

Most projects follow the same flow:

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt` or project config.
3. Run training scripts.
4. Run prediction/evaluation scripts.

Example (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python predict.py
```

For `Fake_News_Detection/`, dependencies are managed via `pyproject.toml`.

## Notes

- Data and trained models are generally kept out of git history.
- Logs, caches, and virtual environments are ignored through the root `.gitignore`.
- Keep each project folder independently runnable.

## Future Improvements

- Add per-project quickstart sections.
- Add experiment tracking (for example, MLflow or Weights & Biases).
- Add a shared utility package for reusable preprocessing components.

## Learnings

1. **Sparse linear models**:

   A **sparse model** works with data where most values are zero.
   - Logistic Regression
     Logistic Regression learns feature weights that push probability toward classes. This algorithm learns word importance, assigns scores, and computes a final score that maps to one of the classes.
     > Logistic Regression → Weighted Probabilistic Voting.
   - Linear SVM
     SVM thinks geometrically. It takes all the features and places them in a space where classes usually group together. Then, SVM tries to separate these classes with a large, safe margin. SVM mainly focuses on hardest boundary points to determine the margin, basically SVM focuses on these edge cases rather than on the regular features.

     > Linear SVM → Geometric Separation.

       <aside>

     _These edge cases are called Support Vectors._

       </aside>

   - Naive Bayes
     Naive Bayes assumes words contribute independently. It learns the frequency and occurrence of words in the article and assigns probabilities based on each word’s importance. This algorithm does not take into account word interactions, grammar, or the phrase structure of the article.
     > NAIVE BAYES → Word Probabilities.

   <aside>

   _SVM and Logistic Regression both are geometric algorithms but SVM is performs well because it thrives on building the optimized separation margin, where as LR works on predicting probabilities._

   </aside>

   Pros:
   - train fast
   - use memory efficiently
   - work surprisingly well for NLP
