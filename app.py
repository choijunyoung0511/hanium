from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import os
import uuid
import tempfile
import requests
from xml.etree import ElementTree as ET

# ---- ChatPDF: 업로드된 PDF별 retriever를 메모리에 보관 ----
_chatpdf_sessions = {}   # { session_id: retriever }

app = Flask(__name__, static_folder="static", static_url_path="")

# ---- 서울시 건설알림이 OPEN API 키 ----
SEOUL_SERVICE_KEY = "756f586a5065647738337a4e77576e"

# ---- 기상청 OPEN API 키 (초단기예보 + 기상특보) ----
WEATHER_SERVICE_KEY = "bf760b16081f482bf9e4640acf9ac58a8a1af754506c38c6610f255c54bfcfd5"

# ---- 카카오 로컬 REST API 키 (좌표->주소 역지오코딩용) ----
KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY", "0289d2c55211ef38b3c9aab4a01918cc").strip()

# ---- AI 시인 LLM (CTransformers / llama-2) ----
_poet_llm = None

def get_poet_llm():
    global _poet_llm
    if _poet_llm is not None:
        return _poet_llm
    try:
        from langchain_community.llms import CTransformers
        _poet_llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q2_k.bin",
            model_type="llama"
        )
        return _poet_llm
    except Exception as e:
        print(f"[POET] LLM 로딩 실패: {e}")
        return None


# =========================================================
# 기상청: 초단기예보 base_date/base_time 계산
# =========================================================
def get_ultra_base_datetime():
    now = datetime.utcnow() + timedelta(hours=9)
    if now.minute < 45:
        now = now - timedelta(hours=1)
    return now.strftime("%Y%m%d"), now.strftime("%H30")


def build_simple_warning(summary: dict) -> str:
    msgs = []
    t1h = summary.get("T1H")
    rn1 = summary.get("RN1")
    sky = summary.get("SKY")
    try:
        if t1h is not None:
            t = float(t1h)
            if t <= -5:
                msgs.append("한파 주의")
            elif t >= 33:
                msgs.append("폭염 주의")
    except Exception:
        pass
    try:
        if rn1 is not None:
            r = float(rn1)
            if r >= 30:
                msgs.append("강한 비(호우) 가능성")
            elif r >= 1:
                msgs.append("비/눈 주의")
    except Exception:
        pass
    if sky == "4":
        msgs.append("짙은 구름·흐림")
    return " · ".join(msgs) if msgs else "특보 없음(간단 판정)"


def fetch_wthr_wrn_list(stn_id: str = "108", days: int = 3):
    now_kst = datetime.utcnow() + timedelta(hours=9)
    to_date = now_kst.strftime("%Y%m%d")
    from_date = (now_kst - timedelta(days=days)).strftime("%Y%m%d")
    url = "http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnList"
    params = {
        "serviceKey": WEATHER_SERVICE_KEY,
        "pageNo": 1, "numOfRows": 50, "dataType": "JSON",
        "stnId": stn_id, "fromTmFc": from_date, "toTmFc": to_date,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") not in (None, "00"):
            return []
        body = data["response"]["body"]
        if int(body.get("totalCount", 0)) == 0:
            return []
        items = body["items"]["item"]
        if isinstance(items, dict):
            items = [items]
        return [{"title": it.get("title"), "tmFc": it.get("tmFc"),
                 "tmSeq": it.get("tmSeq"), "stnId": it.get("stnId")} for it in items]
    except Exception as e:
        print("WthrWrnInfoService error:", e)
        return []


# =========================================================
# 서울 OPEN API 헬퍼
# =========================================================
def seoul_openapi_json(service_name: str, start: int, end: int, *path_parts):
    base = f"http://openapi.seoul.go.kr:8088/{SEOUL_SERVICE_KEY}/json/{service_name}/{start}/{end}"
    tail = "/".join(str(x).strip() for x in path_parts if str(x).strip() != "")
    url = f"{base}/{tail}/" if tail else f"{base}/"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    try:
        return r.json(), url, r.text
    except Exception:
        return None, url, r.text


def seoul_openapi_xml(service_name: str, start: int, end: int, *path_parts):
    base = f"http://openapi.seoul.go.kr:8088/{SEOUL_SERVICE_KEY}/xml/{service_name}/{start}/{end}"
    tail = "/".join(str(x).strip() for x in path_parts if str(x).strip() != "")
    url = f"{base}/{tail}/" if tail else f"{base}/"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    return r.text, url


# =========================================================
# 라우트
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/poet")
def poet_page():
    return render_template("poet.html")


# =========================================================
# API: AI 시인
# =========================================================
@app.route("/api/poet/generate", methods=["POST"])
def poet_generate():
    data = request.get_json(force=True, silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"success": False, "error": "시의 주제를 입력해주세요."}), 400
    llm = get_poet_llm()
    if llm is None:
        return jsonify({"success": False, "error": "AI 시인 모델을 불러오지 못했습니다."}), 503
    try:
        prompt = (
            f"You are a Korean poet. Write a short, beautiful Korean poem about the topic: '{topic}'. "
            f"Use 4-8 lines. Respond with only the poem, no explanation.\n\n시:"
        )
        result = llm.invoke(prompt)
        return jsonify({"success": True, "poem": result.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": f"시 생성 중 오류: {e}"}), 500


# =========================================================
# 날씨 API
# =========================================================
@app.route("/weather")
def weather():
    nx = request.args.get("nx")
    ny = request.args.get("ny")
    if not nx or not ny:
        return jsonify({"success": False, "error": "nx, ny 파라미터가 필요합니다."}), 400

    base_date, base_time = get_ultra_base_datetime()
    ultra_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    ultra_params = {
        "serviceKey": WEATHER_SERVICE_KEY, "dataType": "JSON",
        "numOfRows": 60, "pageNo": 1,
        "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
    }
    try:
        r = requests.get(ultra_url, params=ultra_params, timeout=5)
        r.raise_for_status()
        data = r.json()
        header = data.get("response", {}).get("header", {})
        if header.get("resultCode") not in (None, "00"):
            return jsonify({"success": False, "error": header.get("resultMsg", "기상청 응답 오류")}), 502
        items = data["response"]["body"]["items"]["item"]
        if not items:
            return jsonify({"success": False, "error": "기상 데이터가 비어있습니다."}), 502
        items_sorted = sorted(items, key=lambda x: x.get("fcstTime", "9999"))
        first_time = items_sorted[0]["fcstTime"]
        same_time = [it for it in items_sorted if it.get("fcstTime") == first_time]
        summary = {it.get("category"): it.get("fcstValue") for it in same_time}
        simple_warning = build_simple_warning(summary)
        official_warnings = fetch_wthr_wrn_list(stn_id="108", days=3)
        if official_warnings:
            titles = [w.get("title") for w in official_warnings if w.get("title")]
            warning_text = " / ".join(titles[:3]) if titles else simple_warning
        else:
            warning_text = simple_warning
        return jsonify({
            "success": True,
            "nx": int(nx), "ny": int(ny),
            "baseDate": base_date, "baseTime": base_time, "fcstTime": first_time,
            "T1H": summary.get("T1H"), "RN1": summary.get("RN1"),
            "SKY": summary.get("SKY"), "REH": summary.get("REH"),
            "VEC": summary.get("VEC"), "warning": warning_text,
            "warnings": official_warnings,
        })
    except Exception as e:
        print("weather error:", e)
        return jsonify({"success": False, "error": "기상청 연동 실패"}), 500


# =========================================================
# 카카오 역지오코딩
# =========================================================
@app.route("/kakao/coord2addr")
def kakao_coord2addr():
    lat = request.args.get("lat")
    lng = request.args.get("lng")
    if not lat or not lng:
        return jsonify({"success": False, "error": "lat, lng 파라미터가 필요합니다."}), 400
    if not KAKAO_REST_KEY:
        return jsonify({"success": False, "error": "KAKAO_REST_KEY 미설정"}), 500
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    try:
        r = requests.get(url, headers={"Authorization": f"KakaoAK {KAKAO_REST_KEY}"},
                         params={"x": lng, "y": lat}, timeout=5)
        r.raise_for_status()
        docs = r.json().get("documents", [])
        if not docs:
            return jsonify({"success": True, "address": None})
        doc0 = docs[0]
        road = doc0.get("road_address")
        addr = doc0.get("address")
        name = (road or {}).get("address_name") or (addr or {}).get("address_name")
        return jsonify({"success": True, "address": name})
    except Exception as e:
        print("kakao coord2addr error:", e)
        return jsonify({"success": False, "error": "카카오 역지오코딩 실패"}), 502


# =========================================================
# 당기/차기 작업정보
# =========================================================
@app.route("/pmis/work_terms")
def pmis_work_terms():
    pjt_cd = request.args.get("pjt_cd", "").strip()
    if not pjt_cd:
        return jsonify({"success": False, "error": "pjt_cd 파라미터가 필요합니다."}), 400

    CANDIDATES = [
        "pmisWokInfo", "pmisPjtWkInfo", "pmisPjtWorkInfo",
        "pmisPjtWkTermInfo", "pmisPjtScheduleInfo", "pmisPjtProcInfo",
    ]

    def pick_from_row(row_el, tags):
        for t in tags:
            el = row_el.find(t)
            if el is not None and (el.text or "").strip():
                return (el.text or "").strip()
        return None

    last_err = None
    tried = []

    for svc in CANDIDATES:
        try:
            xml_text, url = seoul_openapi_xml(svc, 1, 10, pjt_cd)
            tried.append(url)
            root = ET.fromstring(xml_text)
            rows = root.findall(".//row")
            if not rows:
                continue
            row0 = rows[0]
            cur_term = pick_from_row(row0, ["NOW_WEEK"])
            cur_cn   = pick_from_row(row0, ["WORK_NOW"])
            nxt_term = pick_from_row(row0, ["NEXT_WEEK"])
            nxt_cn   = pick_from_row(row0, ["WORK_NEXT"])
            current_text = " / ".join([x for x in [cur_term, cur_cn] if x]) or None
            next_text    = " / ".join([x for x in [nxt_term, nxt_cn] if x]) or None
            code_el = root.find(".//CODE")
            msg_el  = root.find(".//MESSAGE")
            return jsonify({
                "success": True, "service": svc, "pjt_cd": pjt_cd,
                "current": current_text, "next": next_text,
                "code": (code_el.text or "").strip() if code_el is not None else None,
                "message": (msg_el.text or "").strip() if msg_el is not None else None,
                "sampleKeys": [child.tag for child in list(row0)[:30]],
            }), 200
        except Exception as e:
            last_err = str(e)

    return jsonify({
        "success": False,
        "error": "당기/차기(XML) 데이터를 찾지 못했습니다.",
        "pjt_cd": pjt_cd, "tried": tried, "last_err": last_err
    }), 200


# =========================================================
# 공정사진 디버그 (실제 태그명 확인용)
# =========================================================
@app.route("/pmis/debug_photos")
def pmis_debug_photos():
    """
    브라우저에서 /pmis/debug_photos?pjt_cd=XXXX 로 접속하면
    실제 XML 태그명과 값을 모두 보여줌
    """
    pjt_cd = request.args.get("pjt_cd", "").strip()
    if not pjt_cd:
        return jsonify({"success": False, "error": "pjt_cd 필요"}), 400
    try:
        xml_text, url = seoul_openapi_xml("pmisPjtPhoto", 1, 10, pjt_cd)
        root = ET.fromstring(xml_text)
        rows = root.findall(".//row")
        raw = []
        for row in rows[:3]:  # 처음 3개만
            raw.append({child.tag: child.text for child in row})
        return jsonify({"success": True, "url": url, "row_count": len(rows), "sample_rows": raw})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


# =========================================================
# 공정사진
# =========================================================
@app.route("/pmis/photos")
def pmis_photos():
    pjt_cd = request.args.get("pjt_cd", "").strip()
    if not pjt_cd:
        return jsonify({"success": False, "error": "pjt_cd 파라미터가 필요합니다."}), 400

    try:
        xml_text, url = seoul_openapi_xml("pmisPjtPhoto", 1, 200, pjt_cd)
        root = ET.fromstring(xml_text)
        rows = root.findall(".//row")

        # ✅ 실제 태그명 콘솔 출력 (디버그용)
        if rows:
            all_tags = {child.tag: (child.text or "").strip() for child in rows[0]}
            print("[DEBUG photo tags]", all_tags)

        photos = []
        for r in rows:
            def t(tag):
                el = r.find(tag)
                return (el.text or "").strip() if el is not None else ""

            # ✅ 실제 태그명 적용 (PIC_URL, CONST_NAME, PHTGRP_DATE)
            photos.append({
                "seq":        t("SEQ"),
                "pjt_cd":     t("PJT_CD"),
                "pjt_name":   t("PJT_NAME"),
                "photo_seq":  t("PHOTO_SEQ"),
                "photo_url":  t("PIC_URL"),
                "photo_name": t("CONST_NAME"),
                "shot_date":  t("PHTGRP_DATE"),
            })

        return jsonify({
            "success": True, "pjt_cd": pjt_cd,
            "count": len(photos), "photos": photos, "source": url
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


# =========================================================
# 디버그: 작업정보 서비스 탐색
# =========================================================
@app.route("/pmis/debug_work_terms")
def pmis_debug_work_terms():
    pjt_cd = request.args.get("pjt_cd", "").strip()
    if not pjt_cd:
        return jsonify({"success": False, "error": "pjt_cd 필요"}), 400

    candidates = [
        "pmisWokInfo", "pmisPjtWkInfo", "pmisPjtWorkInfo", "pmisPjtWkTerm",
        "pmisPjtWkTermInfo", "pmisPjtWkPlan", "pmisPjtWkPlanInfo",
        "pmisPjtSchedule", "pmisPjtScheduleInfo", "pmisPjtProc", "pmisPjtProcInfo",
    ]
    base = f"http://openapi.seoul.go.kr:8088/{SEOUL_SERVICE_KEY}/xml"
    results = []

    for svc in candidates:
        url = f"{base}/{svc}/1/5/{pjt_cd}"
        try:
            r = requests.get(url, timeout=5)
            root = ET.fromstring(r.text)
            rows = root.findall(".//row")
            code_el = root.find(".//CODE")
            msg_el  = root.find(".//MESSAGE")
            sample_keys, sample_pairs = [], []
            if rows:
                row0 = rows[0]
                for child in list(row0)[:20]:
                    sample_keys.append(child.tag)
                    if (child.text or "").strip():
                        sample_pairs.append(f"{child.tag}={(child.text or '').strip()}")
            results.append({
                "service": svc, "http": r.status_code, "rowCount": len(rows),
                "code": (code_el.text or "").strip() if code_el is not None else None,
                "message": (msg_el.text or "").strip() if msg_el is not None else None,
                "sampleKeys": sample_keys, "samplePairs": sample_pairs[:10], "url": url,
            })
        except Exception as e:
            results.append({"service": svc, "error": str(e), "url": url})

    results.sort(key=lambda x: x.get("rowCount", 0), reverse=True)
    return jsonify({"success": True, "pjt_cd": pjt_cd, "results": results})


# =========================================================
# ChatPDF
# =========================================================
@app.route("/chatpdf")
def chatpdf_page():
    return render_template("chatpdf.html")


@app.route("/api/chatpdf/upload", methods=["POST"])
def chatpdf_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "파일이 없습니다."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "PDF 파일만 업로드할 수 있습니다."}), 400
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages  = loader.load_and_split()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        texts    = splitter.split_documents(pages)

        session_id = str(uuid.uuid4())
        persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_{session_id}")
        db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory=persist_dir)
        _chatpdf_sessions[session_id] = db.as_retriever()
        os.unlink(tmp_path)

        return jsonify({
            "success": True, "session_id": session_id,
            "page_count": len(pages), "chunk_count": len(texts),
        })
    except Exception as e:
        print("[ChatPDF upload error]", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chatpdf/ask", methods=["POST"])
def chatpdf_ask():
    data       = request.get_json(force=True, silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    question   = (data.get("question")   or "").strip()

    if not session_id:
        return jsonify({"success": False, "error": "session_id가 없습니다."}), 400
    if not question:
        return jsonify({"success": False, "error": "질문을 입력해주세요."}), 400

    retriever = _chatpdf_sessions.get(session_id)
    if retriever is None:
        return jsonify({"success": False, "error": "PDF 세션을 찾을 수 없습니다. 다시 업로드해주세요."}), 404

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_openai import ChatOpenAI
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import ChatPromptTemplate

        prompt_template = ChatPromptTemplate.from_template("""
다음 문서를 참고하여 질문에 답하세요.
문서:
{context}
질문:
{question}
""")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        result = rag_chain.invoke(question)
        return jsonify({"success": True, "answer": result.content})
    except Exception as e:
        print("[ChatPDF ask error]", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
