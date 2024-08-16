"""
Task 2. 전처리
- ex) 한글 데이터만 남기기
- Okt 같은 형태소 분석기는 사용하지 않아도 좋습니다.
- 띄어쓰기 기준으로 tfidfVectorizer를 적용
"""

import pandas as pd
import re

FILE_PATH = '/home/ubuntu/airflow/dags'

def get_category_sentiment(point):
  if point >= 4: return 0
  elif point >=3 : return 1
  else : return 2


def preprocessing():
   comment_df = pd.read_csv(FILE_PATH + "/data/crawl_df.csv")
   comment_df['target'] = comment_df['point'].apply(get_category_sentiment)
   comment_df.dropna(how='any', subset=['contents'], inplace=True)

   comment_df['preprocessed_comment'] = comment_df['contents'].apply(lambda s : re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", s))
   comment_df.dropna(how='any', subset=['preprocessed_comment'], inplace=True)
   
   comment_df.to_csv(FILE_PATH + "/data/preprocessed_df.csv", index=False)