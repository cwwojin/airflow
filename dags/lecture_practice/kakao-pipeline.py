from datetime import datetime
import json
from airflow import DAG
from pandas import json_normalize
# airflow의 많은 기능 중 기본만 사용

# Operator 등록
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.http.sensors.http import HttpSensor

# 실제 요청과 응답을 수행하는 Operator. 한 번의 요청에 대한 한 번의 응답을 수행할 때 보통 사용
from airflow.providers.http.operators.http import SimpleHttpOperator # 복잡한 내용을 요청하기엔 적합하지 않음 
from airflow.operators.python import PythonOperator # 가장 많이 사용되는 것 중 하나 (중요!!!)
from airflow.operators.bash import BashOperator

REST_API_KEY = "9f214e23cd81651502fd3cf6da2e7e08"
headers = {"Authorization": f"KakaoAK {REST_API_KEY}"}  # 요청 헤더

def _preprocessing(ti):
    # ti: task instance
    #  dag 내의 task의 정보를 얻어 낼 수 있는 객체
    search_result = ti.xcom_pull(task_ids=["extract_kakao"]) # extract_kakao의 결과물 가져오기

    # xcom을 이용해 가지고 온 결과가 아무것도 없다면...
    if not len(search_result):
        raise ValueError("검색 결과가 없습니다.")

    documents = search_result[0]["documents"]
    processed_documents = json_normalize(
        [
            {
                "created_at": document["datetime"],
                "contents": document["contents"],
                "title": document["title"],
                "url": document["url"],
            }
            for document in documents
        ]
    )
        
        # 불러온 데이터를 csv로 저장
    processed_documents.to_csv(
        "/home/ubuntu/airflow/dags/lecture_practice/tmp/processed_result.csv", index=None, header=False
    )

default_args = {"start_date": datetime(2024, 1, 1)}  # 2024년 1월 1일 부터 시작.

# DAG를 구성하기 위한 Task 정의 context 설정
# DAG context 생성
with DAG(
    dag_id="kakao-pipeline", # dag 이름 (고유해야 함)
    schedule_interval="@daily",  # crontab 표현으로 사용 가능 https://crontab.guru/
    default_args=default_args, # dag 초기화 파라미터 생성
    tags=["kakao", "api", "pipeline"], # UI에 표시될 태그명 설
    catchup=False, # Backfill 여부
) as dag:
	#pass

    creating_table = SqliteOperator(
        task_id="creating_table", # 변수의 이름과 똑같이 만들어주는 것이 관행
        sqlite_conn_id="sqlite_con", # 웹 UI에서 만든 커넥션 이름
        sql="""
        CREATE TABLE IF NOT EXISTS kakao_search_result(
            created_at TEXT,
            contents TEXT,
            title TEXT,
            url TEXT
        )
        """ # SQL 작성. 여기서는 데이터를 저장할 테이블을 생성함
    )

    # HTTPSensor: HTTP를 통해 데이터를 수집할 때 접속이 가능한 상태인지 검사
    is_api_available = HttpSensor(
        task_id="is_api_available",
        http_conn_id="kakao_api",
        endpoint="v2/search/web",
        headers=headers,
        request_params={"query": "올림픽"},
        response_check=lambda response: response.json()
    )

    # 실제 HTTP 요청 후 데이터 받아오기
    extract_kakao = SimpleHttpOperator(
        task_id="extract_kakao",
        http_conn_id="kakao_api",
        endpoint="v2/search/web",
        headers=headers,
        data={"query": "올림픽"},
        method="GET", # 요청 방식 설정. GET(기본), POST 등등
        response_filter=lambda res : json.loads(res.text), # res.json() 으로 써도 상관없음
        log_response=True
    )

    preprocess_result = PythonOperator(
        task_id="preprocess_result",
        python_callable=_preprocessing
    )

    # 리눅스 명령어를 Airflow로 실행
    store_result = BashOperator(
        task_id="store_result",
        bash_command='echo -e ".separator ","\n.import /home/ubuntu/airflow/dags/lecture_practice/tmp/processed_result.csv kakao_search_result" | sqlite3 /home/ubuntu/airflow/airflow.db',
    )

    # 모든 과정의 파이프라인화
    creating_table >> is_api_available >> extract_kakao >> preprocess_result >> store_result