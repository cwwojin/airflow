import requests
from bs4 import BeautifulSoup

# 하나의 함수지만 어떤 모듈이나 task로써의 기능을 하고 있다.
def get_naver_finance():
    URL = "https://finance.naver.com/marketindex/"
    response = requests.get(URL)
    page = response.content  # 접속한 웹 사이트의 html 코드를 가져오기
    soup = BeautifulSoup(page, 'html.parser')
    exchange_list = soup.select_one("#exchangeList")
    fin_list = exchange_list.find_all("li")

    datas = []

    for fin in fin_list:
        c_name = fin.select_one("h3.h_lst").text
        exchange_rate = fin.select_one("span.value").text
        change = fin.select_one("span.change").text
        updown = fin.select_one("span.change").nextSibling.nextSibling.text

        data = {
            "c_name": c_name,
            "exchange_rate": exchange_rate,
            "change": change,
            "updown": updown
        }

        datas.append(data)

    # 작업(task)의 결과를 반환. - xcom을 활용하기 위해 리턴
    return datas