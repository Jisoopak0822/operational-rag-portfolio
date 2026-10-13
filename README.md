# Operational RAG Portfolio — Stage 1

학습 가이드 `learning-guide.template(1).md`의 **채널 → QAService → Agent → Retriever → Generator** 구조를 참고해 새로 작성한 학습용 프로젝트입니다. 원본 가이드에 등장하는 운영 코드가 포함된 것은 아닙니다.

## 무엇이 구현됐나?

- **FastAPI**: `/health`, `/chat`, `/v1/chat/completions`(비스트리밍 부분 호환), `/metrics`, 브라우저 UI
- **지식 적재**: YAML frontmatter가 `status: verified` **그리고** `publishable: true`인 Markdown만 검색
- **하이브리드 검색**: 직접 구현한 BM25 + 문자 기반 TF-IDF(기본) → RRF 순위 융합
- **실제 의미 검색(선택)**: sentence-transformers 다국어 모델로 TF-IDF 채널 교체
- **답변 생성(선택)**: OpenAI 호환 chat completion API. 키가 없거나 실패할 때 검색 원문 발췌를 반환
- **평가**: 골든 질문/근거 문서로 Hit@k, Recall@k, MRR@k 측정
- **운영 기초**: Prometheus 기본 메트릭, 요청 지연 측정, 실패 모드 구분
- **자동 테스트**: pytest

> ⚠️ 기본 `tfidf` 모드는 **진짜 임베딩 의미 검색이 아닌 문자 n-gram 기반 어휘 검색**입니다. Milvus/OpenSearch/CrossEncoder/Slack/HyDE/웹 스트리밍은 아직 구현하지 않았습니다. 운영 서비스로 배포하기 위한 인증·접근 제어·문서별 권한 필터도 아직 없습니다. 이 프로젝트는 **로컬 데모**입니다.

## 1. 설치 (macOS / Python 3.11–3.13)

터미널에서 **압축을 푼 `ops_rag_portfolio` 폴더 안에서** 실행하세요.

```bash
cd ops_rag_portfolio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`에서 `LLM_API_KEY`는 **비워 둬도 정상적으로 동작**합니다. 비워두면 챗봇이 답변을 생성하지 않고 검색 원문을 보여줍니다.

## 2. 실행

```bash
python -m uvicorn main:app --reload --port 8000
```

- 웹 챗봇: http://127.0.0.1:8000/
- API 문서: http://127.0.0.1:8000/docs
- 헬스체크: http://127.0.0.1:8000/health
- 메트릭: http://127.0.0.1:8000/metrics

웹에서 **`VPN 오류 720 해결 방법`**이라고 질문해 보세요.

터미널에서 API 호출:

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"VPN 오류 720 해결 방법","top_k":3}'
```

서버 중지: 실행 중인 터미널에서 `Ctrl+C`.

## 3. 검증

```bash
python -m pytest -q
python scripts/run_eval.py --retrieval-only --k 1
python scripts/run_eval.py --retrieval-only --k 3
```

골든셋은 `data/golden.json`입니다. 현재 샘플 수는 매우 작아 점수가 높더라도 일반화된 검색 품질을 입증하지 못합니다. 실사용 문서로 확장한 뒤 동일한 질의 집합에서 전/후 비교하세요.

## 4. 나만의 문서 추가

`data/docs/my_document.md`를 만듭니다.

```markdown
---
title: 내 서비스 사용법
status: verified
publishable: true
---
# 내 서비스 사용법

검증한 사용법 본문을 여기에 입력합니다.
```

로컬 파일을 변경하면 **서버를 재시작**해야 검색 인덱스에 반영됩니다. `draft`이거나 `publishable: false`이거나 메타데이터가 없으면 색인에서 제외합니다. 현재는 단순한 파일 상태 정책이며 사용자별 접근 권한 관리는 없습니다.

## 5. 진짜 의미 검색으로 업그레이드 (선택)

먼저 서버를 중지하고 다음을 설치합니다. 모델은 최초 실행 시 다운로드될 수 있습니다.

```bash
python -m pip install -r requirements-semantic.txt
```

`.env`의 `RAG_EMBEDDING_BACKEND=sentence_transformers`로 수정 후 다시 실행합니다. 한국어와 영어 질의의 의미 유사성 검색을 실험할 수 있습니다. 실제 성능은 데이터셋에서 측정해야 하며 항상 개선되는 것은 아닙니다.

## 6. LLM 연결 (선택, 별도 API 과금 가능)

`.env`의 `LLM_API_KEY`에 유효한 API 키를 넣고 재시작합니다. ChatGPT 구독과 API 사용료는 별도입니다. 생성 실패 시에도 질의 전체를 실패시키지 않고 `mode: fallback`과 검색 발췌를 반환합니다.

## 파일 구조

```text
main.py                  # FastAPI · OpenAI 호환 비스트리밍 API
config.py                # 환경변수
core/qa_service.py       # 공통 질의 서비스·메트릭
rag/agent.py             # 검색 오케스트레이션 경계
rag/retriever_v2.py      # BM25 + TF-IDF / optional embedding + RRF
rag/generator.py         # LLM 생성 / 원문 발췌 폴백
ingest/pipeline.py        # Markdown 검증 · 청킹 · 로컬 로드
web/index.html           # 한 페이지 브라우저 챗봇
data/docs/               # 예제 문서
data/golden.json         # 평가 질문 · 정답 출처
scripts/run_eval.py      # 검색-only 평가
tests/                   # 단위·API 테스트
docs/roadmap.md          # 2~6단계 확장 계획
```

## 아키텍처 특징을 면접에서 설명한다면

"질의 처리와 검색·생성을 별도 계층으로 분리하고, BM25와 TF-IDF 베이스라인을 RRF로 결합했습니다. 원문 검증 상태를 기준으로 색인을 제한했고, 골든셋으로 검색 품질을 LLM과 독립적으로 측정합니다. API 키가 없거나 생성 공급자가 실패할 때 근거 발췌로 전환하는 폴백 경로와 기본 관측 지표도 구현했습니다."

**주의:** 실제 구현·실행·측정한 내용만 포트폴리오 성과로 기재하세요. 앞으로 구축할 Milvus/OpenSearch, CrossEncoder 등의 성과를 구현한 것으로 표현하면 안 됩니다.
