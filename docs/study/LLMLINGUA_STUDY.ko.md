# LLMLingua 공부 노트 (한국어)

> 이 포크에서 공부용으로 정리한 노트입니다. 원저작: Microsoft Research, MIT 라이선스.
> 코드 위치는 이 포크의 `llmlingua/prompt_compressor.py` 기준 줄 번호입니다(upstream 2026-11-06 커밋과 같음).

## 0. 한 줄 요약

**"모델이 쉽게 맞히는 토큰은 정보가 적고, 못 맞히는 토큰에 정보가 있다."**
작은 언어 모델로 토큰마다 "뜻밖인 정도"를 재서, 덜 뜻밖인 토큰부터 지워 프롬프트를 줄인다.

| 버전 | 논문 | 핵심 아이디어 |
| --- | --- | --- |
| LLMLingua | [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) (EMNLP 2023) | 퍼플렉서티로 덩어리 → 토큰 순서로 거칠게·세밀하게 지운다 |
| LongLLMLingua | [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) | **질문 기준**으로 중요도를 잰다. 문서 순위·재배치·문서별 압축률 |
| LLMLingua-2 | [arXiv:2403.12968](https://arxiv.org/abs/2403.12968) | 퍼플렉서티 대신 **분류기**가 토큰마다 남길지 지울지 정한다 |

---

## 1. 바탕 개념: 정보량 = 놀라움

섀넌 정보이론의 **자기 정보량(self-information, surprisal)**:

$$
I(x_i) = -\log p(x_i \mid x_{<i})
$$

- $p(x_i \mid x_{<i})$: 앞 토큰들을 봤을 때 언어 모델이 $x_i$에 준 확률.
- 확률이 높다(예측 쉬움) → $I$가 작다 → 정보가 적다 → 지워도 된다.
- 확률이 낮다(뜻밖) → $I$가 크다 → 정보가 많다 → 남긴다.

**퍼플렉서티(perplexity)**는 이 값의 평균을 지수로 올린 것:

$$
\mathrm{PPL}(x_{1..N}) = \exp\Big(\frac{1}{N}\sum_{i=1}^{N} -\log p(x_i \mid x_{<i})\Big)
$$

코드에서는 토큰별 손실(loss) = $-\log p$ 를 그대로 "토큰의 중요도 점수"로 쓴다.

> 코드: `get_ppl` (165행). `granularity="token"`이면 토큰별 손실 배열, `"sentence"`면 평균 손실을 돌려준다.
> 다음 토큰 예측이므로 logits를 한 칸 밀어(`shift_logits`) 정답 토큰과 맞춘다.

---

## 2. LLMLingua (1세대): 거칠게 → 세밀하게

전체 흐름(`compress_prompt`, 426행):

```text
프롬프트 = 지시문 + 예시/문서들(context) + 질문
  ① 예산 조절기: context 덩어리 중 무엇을 남길지 (덩어리 단위)
  ② 문장 단위 거르기 (선택)
  ③ 반복 토큰 압축: 남은 글을 토큰 단위로 깎음
  ④ (선택) 압축 결과를 원래 단어로 복원
```

### 2-1. 예산 조절기 (Budget Controller)

목표 압축률 $\tau = \dfrac{\text{목표 토큰 수}}{\text{원래 토큰 수}}$.

- 지시문과 질문은 정보 밀도가 높아 **덜** 깎고, 예시·문서(context)는 **더** 깎는다.
- context 덩어리마다 점수를 매겨 높은 순으로 예산이 찰 때까지 담는다.

> 코드: `control_context_budget` (1173행). 덩어리를 점수 순으로 정렬(`get_rank_results`, 1818행)한 뒤
> `target_token -= 덩어리 길이`를 하며 예산이 바닥날 때까지 고른다.

### 2-2. 반복 토큰 압축 (Iterative Token-level Prompt Compression)

한 번에 전체 확률을 계산하면 "지운 토큰 때문에 뒤 토큰의 확률이 바뀌는" 상호의존을 무시하게 된다. 그래서 글을 **구간(segment, 기본 200토큰)** 으로 나누고 앞에서부터:

1. 이미 압축한 앞부분 $\tilde{x}_{<j}$ 를 조건으로 현재 구간의 토큰 손실을 계산

$$
\ell_{j,i} = -\log p\big(x_{j,i} \mid \tilde{x}_{<j},\, x_{j,<i}\big)
$$

2. 구간 목표 비율 $\tau_j$ 를 만족하는 문턱값 $\gamma_j$ 를 고른다(손실을 내림차순 정렬해 상위 $\tau_j$ 비율째 값).
3. $\ell_{j,i} > \gamma_j$ 인 토큰만 남긴다.

$$
\text{keep}(x_{j,i}) \iff \ell_{j,i} > \gamma_j
$$

> 코드: `iterative_compress_prompt` (1523행), 문턱값은 `get_estimate_threshold_base_distribution` (1508행):
> 손실 배열을 정렬해 `int(len * ratio) - 1`번째 값을 문턱으로 쓴다. 구간별 비율은 `get_dynamic_compression_ratio` (1023행).
> 앞 구간의 KV 캐시(`past_key_values`)를 재사용해 계산을 줄인다.

### 2-3. 분포 맞추기 (Distribution Alignment)

작은 모델과 큰 모델(GPT-4 등)은 "무엇이 뜻밖인가"가 다를 수 있다. 그래서 작은 모델을 큰 모델이 만든 글로 지시 튜닝해 확률 분포를 비슷하게 맞춘다. (논문 내용. 라이브러리는 이미 맞춘 모델을 불러 쓰는 구조)

---

## 3. LongLLMLingua: "질문에 비추어" 중요한가

긴 문서·RAG에서는 "언어적으로 뜻밖"보다 **"이 질문에 답하는 데 필요한가"** 가 중요하다.

### 3-1. 문서 순위: 질문의 조건부 퍼플렉서티

문서 $x^{doc}_k$ 를 조건으로 질문 $x^{que}$ (+ 고정 문장 "We can get the answer to this question in the given documents.")의 손실을 잰다.

$$
r_k = -\frac{1}{N_q}\sum_{i} \log p\big(x^{que}_i \mid x^{doc}_k\big)
$$

$r_k$ 가 **낮을수록** 그 문서를 보면 질문이 덜 뜻밖 = 질문과 관련된 문서다. 이 순서로 예산을 채운다.

> 코드: `get_distance_longllmlingua` (2032행)가 `get_condition_ppl` (999행)로 질문을 문서 뒤에 붙여 손실을 계산.
> 순위 방법은 `rank_method`로 BM25, gzip, sentence-BERT, BGE 등과 바꿔 낄 수 있다(같은 함수 안).

### 3-2. 대조 퍼플렉서티 (Contrastive Perplexity)

토큰 $x_i$ 가 질문과 관련 있는지: **질문을 봤을 때** 얼마나 덜 뜻밖이 되는가.

$$
s_i = \underbrace{-\log p(x_i \mid x_{<i})}_{\text{질문 없이}} \;-\; \underbrace{\big(-\log p(x_i \mid x^{que}, x_{<i})\big)}_{\text{질문을 보고}}
$$

$s_i$ 가 클수록 질문이 그 토큰을 예측하는 데 도움이 됐다 = 질문 관련 토큰. 이 값으로 문턱을 정해 남긴다.

> 코드: `condition_compare=True`일 때 `self_loss - loss` (1421행, 1709행).

### 3-3. 그 밖의 장치

- **문서 재배치:** 중요한 문서를 앞·뒤 끝에 둔다(긴 글 가운데를 놓치는 "lost in the middle" 대응). 코드: `reorder_context="two_stage"` (1223행 근처).
- **문서별 동적 압축률:** 순위가 높은 문서는 덜, 낮은 문서는 더 깎는다. 코드: `dynamic_context_compression_ratio` → 순위에 따라 $\pm\delta$ 를 선형 배분.
- **부분열 복원:** 압축 결과에서 잘린 고유명사 등을 원문 단어로 되살린다. 코드: `recover` (1751행).

---

## 4. LLMLingua-2: 분류기로 바꾸다

퍼플렉서티는 **한 방향(왼쪽 문맥만)** 이고 "뜻밖"과 "중요"가 꼭 같지 않다. 2세대는 문제를 **토큰 분류**로 바꾼다.

1. **데이터 증류:** GPT-4에게 "단어를 지우기만 하고 바꾸지 말 것" 규칙으로 원문을 압축시킨다(MeetingBank 회의록).
2. 원문 단어마다 "압축본에 남았나"로 정답(keep / drop)을 붙인다.
3. **양방향 인코더**(XLM-RoBERTa large 또는 mBERT)를 토큰 분류기로 학습한다.

$$
p_{\text{keep}}(x_i) = \mathrm{softmax}(W h_i + b)_{\text{keep}}, \quad h_i = \mathrm{Encoder}(x_{1..N})_i
$$

4. 목표 비율 $\tau$ 에 맞춰 $p_{\text{keep}}$ 상위 토큰만 원래 순서대로 남긴다.

> 코드: `compress_prompt_llmlingua2` (727행). 덩어리 단위 문턱은 `np.percentile(context_probs, 100*(1-rate))` (850행),
> 단어 단위는 `__compress` (2304행)에서 같은 방식(2386, 2407행). 토큰 확률 → 단어 확률은 `token_to_word="mean"`(평균).
> `force_tokens`(반드시 남길 토큰), `force_reserve_digit`(숫자 포함 토큰 강제 보존) 옵션이 있다.

모델: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, 작은 모델 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`.

---

## 5. 세 버전 비교

| | LLMLingua | LongLLMLingua | LLMLingua-2 |
| --- | --- | --- | --- |
| 점수 | 토큰 손실 $-\log p$ | 질문 조건부·대조 손실 | 분류기 $p_{\text{keep}}$ |
| 문맥 | 왼쪽만(인과 LM) | 왼쪽 + 질문 | 양방향 |
| 질문 필요 | 없음 | 있음 | 없음 |
| 속도 | 느림(반복 계산) | 느림 | 빠름(한 번 통과) |
| 약점 | "뜻밖 ≠ 중요"일 때 | 질문이 모호할 때 | 학습 데이터(회의록) 밖 분야 |

---

## 6. 읽는 순서

1. LLMLingua 논문 그림 1과 3절(방법).
2. 코드: `get_ppl`(165) → `control_context_budget`(1173) → `get_estimate_threshold_base_distribution`(1508) → `iterative_compress_prompt`(1523).
3. LongLLMLingua 논문 3절 → 코드 `get_distance_longllmlingua`(2032), `condition_compare` 경로(1421, 1709).
4. LLMLingua-2 논문 데이터 증류 절 → 코드 `compress_prompt_llmlingua2`(727), `__compress`(2304).
5. `examples/` 노트북(RAG, CoT, Code)으로 실제 입출력 확인.

## 7. 용어

| 용어 | 뜻 |
| --- | --- |
| 자기 정보량 (surprisal) | $-\log p$. 뜻밖일수록 큼 |
| 퍼플렉서티 | 평균 surprisal의 지수. 모델이 "헷갈린 정도" |
| 압축률 $\tau$ | 남길 토큰 / 원래 토큰 |
| 문턱값 $\gamma$ | 이 점수보다 높은 토큰만 남김. 정렬로 구함 |
| 조건부 퍼플렉서티 | 다른 글(질문·문서)을 앞에 붙였을 때의 퍼플렉서티 |
| 대조 퍼플렉서티 | 조건 없을 때 − 조건 있을 때. 조건이 얼마나 도움이 됐나 |
| KV 캐시 | 앞부분 계산 결과를 저장해 다음 구간에 재사용 |
