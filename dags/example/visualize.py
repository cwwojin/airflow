import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

FILE_PATH = '/home/ubuntu/airflow/dags'

def visualized():
    comment_df = pd.read_csv(FILE_PATH + '/data/preprocessed_df.csv')
    comment_df = comment_df.dropna(subset=['preprocessed_comment'])

    tfidf_vectorizer = TfidfVectorizer()

    X = tfidf_vectorizer.fit_transform(comment_df['preprocessed_comment'])
    y = comment_df['target']
    
    if hasattr(y, 'toarray'):
        y = y.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with open(FILE_PATH + '/model/rf_clf_with_best_params.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    y_pred = loaded_model.predict(X_test)

    # 혼동 행렬 시각화 및 저장
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(FILE_PATH + '/visualize/confusion_matrix.png')  # 파일로 저장
    plt.show()

    # # ROC 곡선 및 AUC 시각화 및 저장
    # y_prob = loaded_model.predict_proba(X_test)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    # roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig(FILE_PATH + '/visualize/roc_curve.png')  # 파일로 저장
    # plt.show()

    # 특성 중요도 시각화 및 저장
    feature_importances = loaded_model.feature_importances_
    indices = feature_importances.argsort()[::-1]
    features = tfidf_vectorizer.get_feature_names_out()

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(10), feature_importances[indices[:10]], align="center")
    plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=45)
    plt.xlim([-1, 10])
    plt.savefig(FILE_PATH + '/visualize/feature_importances.png')  # 파일로 저장
    plt.show()