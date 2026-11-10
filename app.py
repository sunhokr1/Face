"""AI 관상가 - 얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델로 관상을 풀이하는 Streamlit 앱."""

import base64
import datetime
import hashlib
import io
import os
import threading

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
MAX_SIDE = 1024
MAX_PIXELS = 40_000_000
JPEG_QUALITY = 85
TEMPERATURE = 0.8
MAX_TOKENS = 1500
REQUEST_TIMEOUT = 60  # seconds
SESSION_LIMIT = 10  # 세션(브라우저 탭)당 최대 분석 횟수
DEFAULT_DAILY_LIMIT = 200  # 앱 전체 하루 최대 분석 횟수 (DAILY_LIMIT로 변경 가능)
SECTION_HEADINGS = ["한 줄 관상", "부위별 풀이", "종합 운세", "관상가의 한마디"]
MIN_SECTIONS = 3

DISCLAIMER = "⚠️ 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다."
PRIVACY_NOTICE = "🔒 사진은 분석을 위해 OpenAI API로 전송되며 저장되지 않습니다."
NO_FACE_MESSAGE = "얼굴이 잘 보이는 정면 사진을 올려주세요."

OUTPUT_FORMAT = """\
## 🔮 한 줄 관상
(인상을 한 문장으로)

## 👁️ 부위별 풀이
- **이마 (초년운)**:
- **눈썹·눈 (성품, 대인관계)**:
- **코 (재물운)**:
- **입·입술 (말복, 식복)**:
- **귀 (복)**:
- **턱·얼굴형 (말년운)**:

## 🍀 종합 운세
- **재물운**: ★★★☆☆ (짧은 설명)
- **애정운**: ★★★☆☆ (짧은 설명)
- **건강운**: ★★★☆☆ (짧은 설명)
- **직업운**: ★★★☆☆ (짧은 설명)

## 💬 관상가의 한마디
(긍정적인 조언)"""

SYSTEM_PROMPT = f"""\
당신은 40년 경력의 한국 전통 관상가입니다.
사용자가 보낸 얼굴 사진을 보고 전통 관상학(이마, 눈썹, 눈, 코, 입, 귀, 턱, 얼굴형)의 관점에서
재미있고 긍정적인 관상 풀이를 한국어로 해주세요.

규칙:
- 아래 Markdown 형식을 반드시 지킵니다. 별점은 ★(채움)과 ☆(빈칸)을 합쳐 5개로 표시합니다.
- 오락 목적이므로 따뜻하고 긍정적인 톤을 유지하고, 단점은 조언 형태로 부드럽게 표현합니다.
- 인종, 민족, 종교, 성적 지향, 건강 상태 등 민감한 속성을 추정하거나 언급하지 않습니다.
- 실제 인물이 누구인지 식별하거나 추측하지 않습니다.
- 사진에 사람 얼굴이 없거나 식별이 어려우면 다른 말 없이 "{NO_FACE_MESSAGE}"라고만 답합니다.

출력 형식:
{OUTPUT_FORMAT}"""

USER_PROMPT = "이 사진의 관상을 봐주세요."


class FaceReadingError(Exception):
    """사용자에게 그대로 보여줄 한국어 메시지를 담은 예외."""


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def _get_secret(name: str) -> str | None:
    """st.secrets → 환경변수(.env) 순서로 값을 조회한다."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다 (로컬 환경).
        pass
    return os.environ.get(name)


def get_config() -> tuple[str, str]:
    """(api_key, model)을 반환한다. API 키가 없으면 안내 후 앱을 중단한다."""
    load_dotenv()
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다.\n\n"
            "- 로컬: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=...`를 추가하세요.\n"
            "- Streamlit Cloud: 앱 설정 → Secrets에 `OPENAI_API_KEY`를 등록하세요."
        )
        st.stop()
    model = _get_secret("OPENAI_MODEL") or DEFAULT_MODEL
    return api_key, model


def get_daily_limit() -> int:
    try:
        return int(_get_secret("DAILY_LIMIT") or DEFAULT_DAILY_LIMIT)
    except ValueError:
        return DEFAULT_DAILY_LIMIT


# ---------------------------------------------------------------------------
# 사용량 제한 (공개 배포 시 API 비용 보호)
# ---------------------------------------------------------------------------
@st.cache_resource
def _daily_usage() -> dict:
    """앱 프로세스 전체가 공유하는 하루 사용량 카운터."""
    return {"date": None, "count": 0, "lock": threading.Lock()}


def consume_quota() -> None:
    """분석 1회를 차감한다. 한도를 넘으면 FaceReadingError를 던진다."""
    session_usage = st.session_state.usage
    if session_usage["count"] >= SESSION_LIMIT:
        raise FaceReadingError(
            f"한 번에 최대 {SESSION_LIMIT}회까지 관상을 볼 수 있습니다. 잠시 후 다시 이용해주세요."
        )
    usage = _daily_usage()
    with usage["lock"]:
        today = datetime.date.today()
        if usage["date"] != today:
            usage.update(date=today, count=0)
        if usage["count"] >= get_daily_limit():
            raise FaceReadingError("오늘 준비된 관상 풀이가 모두 소진되었습니다. 내일 다시 찾아주세요.")
        usage["count"] += 1
    session_usage["count"] += 1


# ---------------------------------------------------------------------------
# 이미지 처리 / 분석
# ---------------------------------------------------------------------------
def preprocess_image(file) -> str:
    """이미지를 EXIF 회전 보정·RGB 변환·리사이즈·JPEG 변환 후 base64 문자열로 반환한다."""
    try:
        file.seek(0)
        with Image.open(file) as img:
            # 10MB 이하 PNG도 디코딩 시 수백 MB 메모리를 쓸 수 있으므로 픽셀 수를 먼저 제한
            if img.width * img.height > MAX_PIXELS:
                raise FaceReadingError(
                    "이미지 해상도가 너무 큽니다. 4,000만 화소 이하의 사진을 올려주세요."
                )
            img.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))  # JPEG는 축소 디코딩으로 메모리 절약
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))  # 비율 유지, 긴 변 1024px 이하
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise FaceReadingError(
            "이미지를 읽을 수 없습니다. 손상되지 않은 jpg, png, webp 사진을 올려주세요."
        ) from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_face(client: OpenAI, model: str, image_b64: str) -> str:
    """OpenAI 이미지 분석 모델로 관상 풀이를 요청하고 Markdown 결과를 반환한다."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT,
        )
    except openai.AuthenticationError as exc:
        raise FaceReadingError(
            "OpenAI API 키가 올바르지 않습니다. `.env` 또는 Secrets의 `OPENAI_API_KEY`를 확인해주세요."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise FaceReadingError(
            "이 API 키로는 해당 모델을 사용할 권한이 없습니다. `OPENAI_MODEL` 설정을 확인해주세요."
        ) from exc
    except openai.RateLimitError as exc:
        raise FaceReadingError(
            "요청이 너무 많거나 API 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        ) from exc
    except openai.APITimeoutError as exc:
        raise FaceReadingError("응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.") from exc
    except openai.APIConnectionError as exc:
        raise FaceReadingError("OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.") from exc
    except openai.NotFoundError as exc:
        raise FaceReadingError(
            f"모델 `{model}`을(를) 찾을 수 없습니다. `OPENAI_MODEL` 설정을 확인해주세요."
        ) from exc
    except openai.BadRequestError as exc:
        raise FaceReadingError(
            "요청이 거부되었습니다. 다른 사진으로 시도하거나 이미지 입력을 지원하는 모델인지 확인해주세요."
        ) from exc
    except openai.APIError as exc:
        raise FaceReadingError("관상 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from exc

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if getattr(choice.message, "refusal", None) or not content or NO_FACE_MESSAGE in content[:80]:
        raise FaceReadingError(NO_FACE_MESSAGE)
    # 지정 형식의 섹션이 거의 없으면 관상 풀이가 아닌 응답(거절 문구 등)으로 본다.
    if sum(heading in content for heading in SECTION_HEADINGS) < MIN_SECTIONS:
        raise FaceReadingError(
            "관상 풀이를 만들지 못했습니다. 얼굴이 잘 보이는 다른 사진으로 다시 시도해주세요."
        )
    if choice.finish_reason == "length":
        content += "\n\n---\n*풀이가 길어 일부가 잘렸습니다. 전체 풀이를 보려면 다시 시도해주세요.*"
    return content


# ---------------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Noto+Sans+KR:wght@400;500;700&display=swap');

:root {
  --gs-gold: #d4a74a;
  --gs-gold-light: #ecd08f;
  --gs-ink: #0f1626;
  --gs-panel: #18223a;
  --gs-muted: #b9b2a3;
  --gs-hanji: #f7f0e1;
  --gs-hanji-ink: #2b2418;
  --gs-accent: #8c2f1e;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Noto Sans KR', sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at top, rgba(212, 167, 74, 0.12), transparent 60%),
    var(--gs-ink);
}
/* 고정 footer에 가려지지 않도록 하단 여백 */
[data-testid="stMainBlockContainer"] { padding-bottom: 6rem; }

/* 히어로 */
.gs-hero { text-align: center; padding: 1.5rem 0 0.5rem; }
.gs-hero-icon { font-size: 3rem; line-height: 1; filter: drop-shadow(0 0 12px rgba(212,167,74,.5)); }
.gs-title {
  font-family: 'Gowun Batang', serif; font-weight: 700;
  font-size: clamp(2rem, 7vw, 2.8rem); color: var(--gs-gold-light);
  margin: .5rem 0 .25rem; letter-spacing: .05em;
}
.gs-sub { font-family: 'Gowun Batang', serif; color: var(--gs-muted); font-size: 1.05rem; margin: 0; }
.gs-divider {
  width: 120px; height: 1px; margin: 1.25rem auto;
  background: linear-gradient(90deg, transparent, var(--gs-gold), transparent);
}

/* 사용 방법 */
.gs-step {
  display: flex; align-items: center; gap: .75rem;
  background: var(--gs-panel); border: 1px solid rgba(212,167,74,.25);
  border-radius: 12px; padding: .75rem .9rem;
}
.gs-step-icon { font-size: 1.6rem; }
.gs-step-title { font-weight: 700; color: var(--gs-gold-light); font-size: .95rem; }
.gs-step-desc { color: var(--gs-muted); font-size: .8rem; }

/* 입력 카드 */
.st-key-input_card {
  background: var(--gs-panel); border: 1px solid rgba(212,167,74,.3);
  border-radius: 16px; padding: 1rem 1.1rem;
}

/* 미리보기 */
.gs-preview { text-align: center; margin: .5rem 0 .25rem; }
.gs-preview img {
  width: 100%; max-width: 350px; border-radius: 14px;
  border: 2px solid var(--gs-gold); box-shadow: 0 8px 28px rgba(0,0,0,.45);
}
.gs-preview-caption { color: var(--gs-muted); font-size: .85rem; margin-top: .4rem; }

/* 관상 보기 버튼 */
.st-key-analyze_btn button {
  background: linear-gradient(135deg, #e6c06a, #b8862f);
  color: var(--gs-hanji-ink); border: none; border-radius: 12px;
  min-height: 3.2rem; box-shadow: 0 4px 18px rgba(212,167,74,.35);
}
.st-key-analyze_btn button p { font-size: 1.15rem; font-weight: 700; color: var(--gs-hanji-ink); }
.st-key-analyze_btn button:hover { filter: brightness(1.08); }

/* 결과 카드 (한지) */
.st-key-result_card {
  background: var(--gs-hanji); border: 1px solid var(--gs-gold);
  border-radius: 16px; padding: 1.4rem 1.5rem; margin-top: 1rem;
  box-shadow: 0 10px 36px rgba(0,0,0,.4);
}
.st-key-result_card, .st-key-result_card p, .st-key-result_card li,
.st-key-result_card strong, .st-key-result_card span { color: var(--gs-hanji-ink); }
.st-key-result_card h2 {
  font-family: 'Gowun Batang', serif; font-size: 1.3rem; color: var(--gs-accent);
  border-bottom: 1px solid rgba(140,47,30,.25); padding: .9rem 0 .35rem; margin-bottom: .5rem;
}
.st-key-result_card li { margin-bottom: .3rem; line-height: 1.7; }
.st-key-result_card [data-testid="stHeaderActionElements"] { display: none; }
.gs-result-title {
  text-align: center; font-family: 'Gowun Batang', serif; font-size: 1.1rem;
  color: var(--gs-accent); letter-spacing: .3em;
}

/* 고정 footer */
.gs-footer {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 100;
  background: rgba(15,22,38,.92); backdrop-filter: blur(6px);
  border-top: 1px solid rgba(212,167,74,.25);
  text-align: center; padding: .45rem 1rem; font-size: .72rem; color: var(--gs-muted); line-height: 1.5;
}

@media (max-width: 640px) {
  .st-key-result_card { padding: 1.1rem 1rem; }
  .gs-sub { font-size: .95rem; }
}
</style>
"""

STEPS = [
    ("📸", "사진 올리기", "정면 얼굴 사진을 올려요"),
    ("🔮", "관상 보기", "버튼을 누르면 풀이 시작"),
    ("📜", "결과 확인", "부위별 풀이와 운세 확인"),
]

TIPS = """\
### 📌 좋은 사진 고르는 팁
- 얼굴이 **정면**을 향한 사진
- **밝은 조명**에서 찍은 사진
- **안경·모자·마스크**는 벗고
- 머리카락이 이마와 귀를 가리지 않게
- 한 사람만 나온 사진
"""


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
EMPTY_ANALYSIS = {"image_hash": None, "image_b64": None, "result": None, "error": None}


def init_state() -> None:
    # 분석 상태는 하나의 dict에 담아 "제자리 수정"만 한다.
    # 분석 중 버튼을 다시 누르면 Streamlit이 새 실행을 겹쳐 시작하는데, 이때 중단된 실행이
    # st.session_state에 *대입*한 값은 다음 실행에 반영되지 않는다. 같은 dict 객체를
    # 수정하면 모든 실행이 같은 상태를 본다.
    st.session_state.setdefault("analysis", dict(EMPTY_ANALYSIS))
    st.session_state.setdefault("analysis_lock", threading.Lock())
    st.session_state.setdefault("usage", {"count": 0})  # 제자리 수정용 dict (analysis와 같은 이유)
    st.session_state.setdefault("input_version", 0)  # 위젯 key를 바꿔 업로드/촬영 초기화


def get_analysis() -> dict:
    return st.session_state.analysis


def run_analysis(api_key: str, model: str) -> None:
    """관상 분석을 실행하고 결과/오류를 분석 상태에 저장한다.

    세션별 잠금으로 API 호출이 한 번만 일어나게 하고, 겹쳐 시작된 다른 실행은 끝날 때까지 기다린다.
    """
    state = get_analysis()
    lock = st.session_state.analysis_lock
    if not lock.acquire(blocking=False):
        # 다른 실행이 분석 중 → 그 결과를 기다렸다가 표시
        if lock.acquire(timeout=REQUEST_TIMEOUT + 10):
            lock.release()
        return
    try:
        if state["result"] is not None:  # 대기 중 다른 실행이 이미 완료
            return
        image_hash, image_b64 = state["image_hash"], state["image_b64"]
        try:
            consume_quota()
            client = OpenAI(api_key=api_key)
            result, error = analyze_face(client, model, image_b64), None
        except FaceReadingError as exc:
            result, error = None, str(exc)
        # 분석 중 사용자가 다른 사진으로 바꿨다면 이전 사진의 결과는 버린다.
        if state["image_hash"] == image_hash:
            state.update(result=result, error=error)
    finally:
        lock.release()


def reset_state() -> None:
    get_analysis().update(EMPTY_ANALYSIS)
    st.session_state.input_version += 1


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="gs-hero">'
        '<div class="gs-hero-icon">🔮</div>'
        '<div class="gs-title" role="heading" aria-level="1">AI 관상가</div>'
        '<p class="gs-sub">얼굴에 새겨진 당신의 운명을 읽어드립니다</p>'
        "</div>"
        '<div class="gs-divider"></div>',
        unsafe_allow_html=True,
    )
    for col, (icon, title, desc) in zip(st.columns(3), STEPS):
        col.markdown(
            f'<div class="gs-step"><span class="gs-step-icon">{icon}</span>'
            f'<div><div class="gs-step-title">{title}</div>'
            f'<div class="gs-step-desc">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f'<div class="gs-footer">{DISCLAIMER}<br>{PRIVACY_NOTICE}</div>',
        unsafe_allow_html=True,
    )


INPUT_UPLOAD = "📁 사진 업로드"
INPUT_CAMERA = "📷 카메라 촬영"


def get_input_image():
    """선택한 입력 방식(업로드/카메라)의 이미지를 반환한다.

    st.tabs는 숨은 탭의 위젯도 렌더링해서 페이지 로드 즉시 카메라 권한을 요청하므로,
    선택한 방식의 위젯만 렌더링한다.
    """
    version = st.session_state.input_version
    with st.container(key="input_card"):
        mode = st.segmented_control(
            "입력 방식",
            [INPUT_UPLOAD, INPUT_CAMERA],
            default=INPUT_UPLOAD,
            required=True,
            key="input_mode",
            label_visibility="collapsed",
        )
        if mode == INPUT_CAMERA:
            return st.camera_input("카메라로 촬영하기", key=f"camera_{version}")
        return st.file_uploader(
            "얼굴이 잘 보이는 정면 사진을 올려주세요 (최대 10MB)",
            type=ALLOWED_TYPES,
            key=f"uploader_{version}",
        )


def render_preview(image_b64: str) -> None:
    st.markdown(
        f'<div class="gs-preview"><img src="data:image/jpeg;base64,{image_b64}" alt="분석할 사진">'
        '<div class="gs-preview-caption">분석할 사진</div></div>',
        unsafe_allow_html=True,
    )


def render_result(result: str) -> None:
    with st.container(key="result_card"):
        st.markdown('<div class="gs-result-title">관 상 풀 이</div>', unsafe_allow_html=True)
        st.markdown(result)


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    init_state()
    render_header()
    render_footer()
    with st.sidebar:
        st.markdown(TIPS)

    api_key, model = get_config()

    image_file = get_input_image()
    if image_file is None:
        return

    if image_file.size > MAX_UPLOAD_BYTES:
        st.warning("사진 용량이 10MB를 초과합니다. 더 작은 사진을 올려주세요.")
        return

    state = get_analysis()
    image_hash = hashlib.sha256(image_file.getvalue()).hexdigest()
    if image_hash != state["image_hash"]:
        # 새 이미지 → 전처리 1회 수행, 이전 결과 초기화
        try:
            image_b64, error = preprocess_image(image_file), None
        except FaceReadingError as exc:
            image_b64, error = None, str(exc)
        state.update(image_hash=image_hash, image_b64=image_b64, result=None, error=error)

    if state["image_b64"] is None:
        st.error(state["error"])
        st.button("🔄 다시 하기", on_click=reset_state, width="stretch", key="reset_btn")
        return

    render_preview(state["image_b64"])

    has_outcome = state["result"] is not None or state["error"] is not None
    # 결과가 있으면 버튼을 숨겨 중복 호출 방지. 오류 시에는 재시도할 수 있게 버튼 유지.
    if state["result"] is None and st.button(
        "🔮 관상 보기", type="primary", width="stretch", key="analyze_btn"
    ):
        with st.spinner("관상을 보는 중입니다..."):
            run_analysis(api_key, model)
        st.rerun()

    if state["error"]:
        st.warning(state["error"])
    if state["result"]:
        render_result(state["result"])

    if has_outcome:
        st.button("🔄 다시 하기", on_click=reset_state, width="stretch", key="reset_btn")


if __name__ == "__main__":
    main()
