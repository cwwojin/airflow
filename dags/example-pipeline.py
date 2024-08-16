from datetime import datetime
from airflow import DAG

# 직접 만든 파이썬 모듈 불러오기
from airflow.operators.python import PythonOperator

from pandas import json_normalize

from example.crawler import crawl_data
from example.preprocessing import preprocessing
from example.modeling import modeling
from example.final import final_train
from example.visualize import visualized

defalut_args = {
    "start_date": datetime(2024, 1, 1)
}

with DAG(
    dag_id="example-pipeline",
    schedule_interval="@daily",
    default_args=defalut_args,
    tags=["kakao", "review", "example"],
    catchup=False
) as dag:
    
    crawl_result = PythonOperator(
        task_id="crawl_result",
        python_callable=crawl_data
    )

    preprocessing_result = PythonOperator(
        task_id="preprocessing_result",
        python_callable=preprocessing
    )

    modeling_result = PythonOperator(
        task_id="modeling_result",
        python_callable=modeling
    )

    final_result = PythonOperator(
        task_id="final_result",
        python_callable=final_train
    )

    visualize_result = PythonOperator(
        task_id="visualize_result",
        python_callable=visualized
    )

    crawl_result >> preprocessing_result >> modeling_result >> final_result >> visualize_result