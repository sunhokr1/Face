# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델이 전통 관상학 관점으로 풀이해 주는 Streamlit 앱입니다.

> 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다.

## 로컬 실행

Python 3.10 이상이 필요합니다. (Streamlit Cloud 배포 시 Advanced settings에서 Python 3.12 선택 권장)

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
streamlit run app.py
```

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 푸시합니다. (`.env`는 `.gitignore`로 제외됩니다.)
2. [share.streamlit.io](https://share.streamlit.io)에서 저장소, 브랜치, `app.py`를 선택합니다.
3. **Advanced settings → Secrets**에 다음을 입력합니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   DAILY_LIMIT = "200"   # 앱 전체 하루 최대 분석 횟수 (선택)
   ```

4. Deploy를 누릅니다.

## 비용 보호

- 세션(브라우저 탭)당 최대 10회, 앱 전체 하루 최대 `DAILY_LIMIT`회(기본 200)까지만 분석합니다.
- 앱 내 카운터는 앱이 재시작되면 초기화되므로, **OpenAI 대시보드(Settings → Limits)에서 월 사용 한도도 반드시 설정**하세요.
