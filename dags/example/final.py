"""
Task 4. 재 학습
- Task 3에서 찾아낸 하이퍼 파라미터를 토대로 훈련 세트를 사용한 모델 훈련
- 테스트 세트로 일반화 오차 확인
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FILE_PATH = '/home/ubuntu/airflow/dags'

def final_train():
    comment_df = pd.read_csv(FILE_PATH + '/data/preprocessed_df.csv')
    param_df = pd.read_csv(FILE_PATH + '/data/grid_search_results.csv')

    comment_df = comment_df.dropna(subset=['preprocessed_comment'])
    best_params = param_df.iloc[0].drop('best_score').to_dict()

    tfidf_vectorizer = TfidfVectorizer()

    X = tfidf_vectorizer.fit_transform(comment_df['preprocessed_comment'])
    y = comment_df['target']

    if hasattr(y, 'toarray'):
        y = y.toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(**best_params)
    rf_clf.fit(X_train, y_train)

    with open(FILE_PATH + '/model/rf_clf_with_best_params.pkl', 'wb') as model_file:
        pickle.dump(rf_clf, model_file)

    with open(FILE_PATH + '/model/tfidf_vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)