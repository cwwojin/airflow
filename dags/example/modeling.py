"""
Task 3. 모델링
- 실습에서는 LogisticRegression만 사용했지만 앙상블 모델로도 수행이 가능
- 할 수 있으시면 GridSearch 수행하고, 최적의 하이퍼 파라미터를 csv로 저장
- 이 때 들어가는 데이터는 모든 데이터를 다 사용하는 것으로
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os.path as path


def modeling():
    comment_df = pd.read_csv("./data/preprocessed_df.csv")
    comment_df = comment_df.dropna(subset=["preprocessed_comment"])
    tfidf_vectorizer = TfidfVectorizer()

    X = tfidf_vectorizer.fit_transform(comment_df["preprocessed_comment"])
    y = comment_df["target"]

    rf_clf = RandomForestClassifier()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    grid_search = GridSearchCV(
        estimator=rf_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )

    grid_search.fit(X, y)

    # Best model
    best_rf_clf = grid_search.best_estimator_

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    results_df = pd.DataFrame([best_params])
    results_df["best_score"] = best_score

    results_df.to_csv(
        "./data/grid_search_results.csv",
        index=False,
    )
