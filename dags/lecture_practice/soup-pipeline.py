from datetime import datetime
from airflow import DAG

# 직접 만든 파이썬 모듈 불러오기
from lecture_practice.naver_scraper.naver_example import get_naver_finance
from airflow.operators.python import PythonOperator

from pandas import json_normalize

defalut_args = {
    "start_date": datetime(2024, 1, 1)
}

def _processing_result(ti):
    '''
    :param ti: task instance
    '''
    # search_result = None
    # task instance를 이용해 특정 Operator에서 데이터를 받아옴
    search_result = ti.xcom_pull(task_ids=["soup_result"])
    if not len(search_result):
        raise ValueError("검색 결과가 없습니다.")

    processed_documents = json_normalize(search_result)
    print(search_result)
    processed_documents.to_csv("/home/ubuntu/airflow/dags/lecture_practice/tmp/naver_result.csv", index=None, header=False)

with DAG(
    dag_id="naver-finance-pipeline",
    schedule_interval="@hourly",
    default_args=defalut_args,
    tags=["naver", "finance", "soup"],
    catchup=False
) as dag:
    
    # 데이터 수집 task 실행
    soup_result = PythonOperator(
        task_id="soup_result",
        python_callable=get_naver_finance
    )

    process_naver_finance = PythonOperator(
        task_id="process_naver_finance",
        python_callable=_processing_result
    )

    soup_result >> process_naver_finance