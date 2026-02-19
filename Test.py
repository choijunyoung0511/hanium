import requests
import xml.etree.ElementTree as ET

# 1) 본인 인증키 넣기
SERVICE_KEY = "477846527365647734336b63744d68"

# 2) URL 구성 (1~40건만 예시로 가져오기)
url = f"http://openapi.seoul.go.kr:8088/{SERVICE_KEY}/xml/pmisPjtList/1/40/"

# 3) 요청 보내기
res = requests.get(url)
res.raise_for_status()   # 문제 있으면 에러 발생

xml_data = res.content   # bytes

# 4) XML 파싱
root = ET.fromstring(xml_data)   # <PROJECT_LIST> 루트

# 5) row 태그들 가져오기
rows = root.findall("row")

projects = []
for row in rows:
    def get(tag):
        elem = row.find(tag)
        return elem.text.strip() if elem is not None and elem.text else None

    project = {
        "seq": get("SEQ"),
        "code": get("PJT_CD"),
        "name": get("PJT_NAME"),
        "facility_type": get("FCT_F6_NM"),           # 도로 / 도시철도 / 공원 등
        "gu_name": get("GU_NAME"),                   # 자치구
        "begin_date": get("PJT_BGN1_DATE"),          # 착공일
        "end_date": get("PJT_COMPL_PREARR_DATE"),    # 준공예정일
        "amount": get("TOT_CNTRT_AMT"),              # 공사비(억 단위)
        "scale": get("PJT_SCALE"),                   # 사업규모 설명
    }
    projects.append(project)

# 6) 테스트 출력
print(len(projects), "건")
for p in projects[:3]:   # 앞 3개만 보기
    print(p)
