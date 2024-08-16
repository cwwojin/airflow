"""
Task 1. 크롤링
- 리뷰 데이터
- 지하철역 2개 정도만 선정 → 해당 지하철역의 반경 1km 맛집 찾기 - > 각 맛집의 리뷰 수
- 모듈화 시켜주기
"""
import requests
import pandas as pd

REST_API_KEY = "9f214e23cd81651502fd3cf6da2e7e08"
headers = {"Authorization": f"KakaoAK {REST_API_KEY}"}  # 요청 헤더

KEYWORD_LOCAL_URL = "https://dapi.kakao.com/v2/local/search/keyword.json?query={}&radius=1000"
COMMENT_URL = "https://place.map.kakao.com/m/commentlist/v/{}/{}?order=USEFUL&onlyPhotoComment=false"

FILE_PATH = '/home/ubuntu/airflow/dags'

def crawl_data():
  keywords = ["강남역", "홍대입구역"]

  all_reviews = []
  
  for key in keywords: # 각 지하철 역에 대해
    response = requests.get(KEYWORD_LOCAL_URL.format(key + " 맛집"), headers=headers)
    
    stores = response.json().get('documents', [])

    for store in stores:
      store_id = store['id']      
      all_comments = []
      comment_id = 0 # 첫 번째 코멘트의 id는 무조건 0
      has_next = True # has_next가 true면 계속 수집을 한다. true라면 마지막 코멘트id를 새롭게 넣어서 크롤링한다.
      
      while has_next:
        # 수집해야 할 URL 생성
        SCRAP_COMMENT_URL = COMMENT_URL.format(store_id, comment_id)
        response = requests.get(SCRAP_COMMENT_URL) # 만들어진 url로 요청

        if not response.json():
          break
        
        comment_datas = response.json()['comment']
        
        # 댓글 데이터 가져오기
        comment_list = comment_datas.get('list', [])
        all_comments.extend(comment_list)
        
        # 다음 페이지 존재 여부 확인
        has_next = comment_datas.get('hasNext', False)

        # has_next가 True인 경우 마지막 코멘트의 id를 comment_id로 설정
        if has_next:
          comment_id = comment_list[-1]['commentid']

      all_reviews.extend(all_comments) 

  comment_df = pd.DataFrame(all_reviews)
  comment_df.to_csv(FILE_PATH + "/data/crawl_df.csv", index=False)