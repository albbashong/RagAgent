#  RAG 기반 코드 분석·리팩토링 Agent

##  시스템 구조
<img width="789" src="https://github.com/user-attachments/assets/82d8ea84-41a7-43ac-83d2-f437f6ecb4c7" />

---

# 주요 기능

## 1) 사용자 의도 파악 (Intent Classification)

사용자의 자연어 요청을 분석해 **필요한 작업 유형을 자동 분류**

- 일반 대화
- DB 조회  
- 로컬 파일 접근  
- 웹 검색

등을 LLM이 판단하여 적절한 함수 호출을 자동 선택

➡️ **사용자는 자연스러운 대화만으로 필요한 함수를 호출**

---

## 2) 자연어 기반 DB 조회 (RAG 기반 Retrieval)

GitHub 프로젝트나 코드 내용에 대한 질문이 들어오면 아래 파이프라인이 동작.

1. 사용자 질의를 **Embedding Vector**로 변환  
2. DB에 저장된 문서 embeddings와 **유사도 계산**  
3. Top-k 관련 chunk 추출  
4. 사용자 질의 + 관련 Context를 LLM 입력으로 전달  
5. 정확한 해설·요약·코드 분석 제공

➡️ **사용자는 직접 파일을 찾을 필요 없이 자연어로 원하는 정보를 조회**

---

## 3) 코드 리팩토링 (LLM 기반 Code Transformer)

Agent는 검색된 Context와 사용자 요청을 기반으로 코드를 리팩토링

- 코드 구조 분석  
- 모듈·함수 간 역할 정리  
- 개선점 제안  
- 프레임워크 변환 (TensorFlow → PyTorch 등)  
- 필요 시 함수 단위 재작성  

➡️ **협업 프로젝트 온보딩 속도를 높이고 코드 품질을 유지하는 데 도움**

---

#  Routing Signal Breakdown

Agent는 아래 세 가지 신호를 종합해 Routing을 결정

| Signal Type | Description | Output Example |
|-------------|-------------|----------------|
| **Intent Router** | 사용자 의도 분류 | `intent="read"`, `confidence=0.83` |
| **Explicit Signal** | run.py, SELECT 등 텍스트 단서 기반 | `source="postgres"`, `confidence=0.95` |
| **Conversation Continuity** | 이전 대화 문맥 embedding 기반 | `reuse="repo_chunks"`, `similarity=0.72` |

---

## Self-Checker Output  
| Metric | 설명 | 예시 |
|--------|------|------|
| **Confidence** | 모델이 해당 답변에 대해 가진 확신도 | `0.41` |
| **Freshness Need** | 최신 정보 필요 여부 | `yes` |

---

#  Routing 방식

Routing Aggregator는 아래와 같은 **가중합(Weighted Sum)** 으로 신호 통합

```
 final_confidence =
 source_confidence * 0.4 +
 intent_confidence * 0.4 +
 self_check_conf * 0.2
```
이 결과를 기준으로:

- 최종 source (repo_chunks / postgres / filesystem / web / direct)
- 실행될 function 호출  
- fallback (유사 질의 차단 등)

로 최종함수 선택

---

#  요약

이 시스템은 다음을 목표를 바탕으로 설계

- 자연어 기반 코드 탐색  
- RAG 기반 파일 검색 및 문맥 분석  
- LLM 기반 코드 리팩토링 자동화  
- 다중 라우팅 신호 기반의 정확한 함수 호출  

#  전체 Routing Flow
<img width="778" height="430" alt="image" src="https://github.com/user-attachments/assets/b048b7de-332d-4612-bb81-913883d5c8e3" />




