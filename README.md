# chart_ml_v4 — 미국 주식 5분봉 ML 예측 모델

XGBoost + 1D CNN 앙상블 기반 5분봉 방향 예측 모델.  
47개 종목, 거시지표 + 섹터 더미 + 상호작용 피처를 활용해 다음 봉의 상승/하락/횡보를 예측한다.

---

## 성능 요약 (v4 최신)

| 지표 | 값 |
|------|-----|
| Hold-out 정확도 | 80.4% |
| 합산 수익률 (60%+) | +196.83% |
| 매수 단독 (60%+) | +98.72% |
| 매도 단독 (60%+) | +98.11% |
| 회귀 방향 일치율 (60%+) | 83.9% |
| 테스트 샘플 | 40,213개 |
| Buy & Hold | +6.77% |

> 백테스트 기간 약 12 영업일 (Yahoo Finance 5분봉 최대 60일 기준, 테스트셋 20%)

### 실전 수익률 추정 (1억 기준, 하루 2~3신호 선택)

| 시나리오 | 12일 수익 | 연환산 |
|----------|-----------|--------|
| 낙관적 | 2,000~3,000만원 | ~300% |
| 현실적 | 1,000~1,500만원 | ~150% |
| 보수적 | 500~800만원 | ~80% |

> 슬리피지/과적합 감안 50% 할인 적용. 증권사 API 자동매매 시 백테스트 수치에 가까워질 수 있음.

---

## 피처 중요도 Top 15 (최신)

| 순위 | 피처 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | atr_ratio | 0.134 | 변동성 비율 (압도적 1위) |
| 2 | sector_volatile | 0.100 | 고변동성 섹터 더미 |
| 3 | bb_width | 0.060 | 볼린저 밴드 폭 |
| 4 | session_open | 0.048 | 장 시작 30분 |
| 5 | vix_x_volatile | 0.030 | VIX × 고변동성 상호작용 |
| 6 | mins_to_close | 0.030 | 마감까지 남은 시간 |
| 7 | dxy_signal | 0.030 | 달러 신호 |
| 8 | mins_from_open | 0.024 | 장 시작 후 경과 시간 |
| 9 | sector_semicon | 0.020 | 반도체 섹터 더미 |
| 10 | session_morning | 0.018 | 오전 세션 |

---

## 버전 히스토리

| 버전 | 주요 변경 | 정확도 | 합산 수익률 |
|------|-----------|--------|-------------|
| v1 | 기본 XGBoost, 5종목 | ~70% | - |
| v2 | 피처 확장, 20종목 | 83.9% | +25.51% |
| v3 | CNN 추가, VIX, 30종목 | 84.3% | +38.94% |
| v4-1차 | 47종목, 거시지표 4개, 섹터더미, 상호작용 4개 | 79.6% | +151.25% |
| v4-2차 | Optuna 패널티 버그 수정 | 78.2% | +16.11% |
| v4-3차 | Optuna 패널티 수정 완료 | 79.8% | +145.62% |
| v4-4차 | 피처 정리 13개 제거, 상호작용 8개 추가, seed=42 고정 | **80.4%** | **+196.83%** |

---

## 폴더 구조

```
chart_ml_v4/
├── data_pipeline_5m.py   # 데이터 수집 & 피처 생성
├── train_model.py        # XGBoost + CNN + Optuna 학습
├── predict.py            # 실시간 예측 (단일 종목)
├── visualize_model.py    # 학습 결과 시각화 (백테스트 리포트)
├── visualize_predict.py  # 예측 차트 시각화
├── output/               # CSV, CNN 시퀀스 (.npy)
└── models/               # 저장된 모델 파일
    ├── chart_model.json       # XGBoost 분류 모델
    ├── chart_model_reg.json   # XGBoost 회귀 모델 (변동폭 예측)
    ├── cnn_model.pt           # 1D CNN 모델
    ├── feature_list.txt       # 피처 목록
    ├── label_classes.json     # 라벨 클래스
    └── best_params.json       # Optuna 최적 파라미터
```

---

## 실행 순서

### 1. 데이터 수집
```bash
python data_pipeline_5m.py
```
- 47개 종목 × 60일 5분봉 수집
- 거시지표 (VIX, 금, 유가, DXY) 자동 수집
- output/combined_5m_features.csv 생성

### 2. 모델 학습
```bash
python train_model.py --optuna-trials 100 --cnn-epochs 25
```
- Optuna 100회 하이퍼파라미터 탐색 (seed=42 고정)
- XGBoost + 1D CNN 앙상블 학습
- XGBoost 회귀 모델 추가 학습 (변동폭 예측)

### 3. 예측
```bash
python predict.py --ticker AAPL
```

### 4. 시각화
```bash
py -3.12 visualize_model.py
py -3.12 visualize_predict.py --ticker AAPL --bars 50
```

---

## 종목 구성 (47개)

| 섹터 | 종목 |
|------|------|
| 테크 (9) | AAPL, MSFT, GOOGL, META, AMZN, NFLX, ORCL, CRM, UBER |
| 반도체 (7) | NVDA, AMD, INTC, QCOM, AVGO, MU, TSM |
| 고변동성 (6) | TSLA, COIN, PLTR, IREN, APP, IONQ |
| 금융 (7) | JPM, BAC, GS, V, MS, SCHW, BLK |
| 헬스케어 (5) | JNJ, UNH, LLY, MRNA, ABBV |
| 에너지 (4) | XOM, CVX, OXY, SLB |
| ETF (3) | SPY, QQQ, IWM |
| 소비재 (5) | WMT, COST, NKE, SHOP, KO |
| 산업재 (1) | CAT |

---

## 피처 구성

### 기술 지표
- **캔들**: body, upper/lower_wick, body_ratio, wick_ratio, is_bullish, gap_pct, streak
- **패턴**: pat_bullish_engulf, pat_bearish_engulf
- **이동평균**: dist_ma9/20/50/200, ma9_above_ma20, ma20_above_ma50, ma20/50_slope, above_ma20
- **VWAP**: dist_vwap, above_vwap, vwap_cross_up/dn
- **모멘텀**: rsi14, macd_hist, macd_above_signal, macd_hist_rising, stoch_k, stoch_overbought/oversold, roc_1/3/6/12
- **거래량**: vol_ratio_1h/4h, vol_spike, cum_vol_today, obv_slope, bull_vol, bear_vol
- **변동성**: bb_width, bb_pct, bb_above_upper, bb_below_lower, atr_ratio
- **지지저항**: pos_in_range_1h/4h
- **래그**: return_lag1~6, rsi_lag1/3, macd_hist_lag1

### 세션 피처
- mins_from_open, mins_to_close
- session_open, session_morning, session_afternoon, session_close

### 거시지표 (9개)
- VIX: vix_change, vix_signal, vix_close
- 금: gold_change, gold_signal
- 유가: oil_change, oil_signal
- 달러: dxy_change, dxy_signal

### 섹터 더미 (9개)
- sector_tech/semicon/volatile/finance/health/energy/etf/consumer/industrial

### 상호작용 피처 (12개)

| 피처 | 의미 |
|------|------|
| oil_x_energy | 유가 × 에너지주 |
| oil_x_semicon | 유가 × 반도체 (음의 상관) |
| oil_x_consumer | 유가 × 소비재 |
| dxy_x_tech | 달러 × 기술주 |
| dxy_x_energy | 달러 × 에너지 |
| dxy_x_semicon | 달러 × 반도체 |
| vix_x_volatile | VIX × 고변동성 |
| vix_x_finance | VIX × 금융주 |
| vix_x_health | VIX × 헬스케어 |
| vix_x_etf | VIX × ETF |
| gold_x_volatile | 금 × 고변동성 |
| gold_x_finance | 금 × 금융주 |

---

## 모델 구조

```
입력 피처 (~90개)
      ↓
XGBoost 분류 (방향 예측: 상승/횡보/하락)
XGBoost 회귀 (변동폭 예측: future_return_pct)
1D CNN      (시계열 패턴 학습)
      ↓
앙상블 (Optuna 가중치 최적화)
      ↓
출력: 방향 + 신뢰도 + 예상 변동폭 + 예상 종가
```

### Optuna 목적함수
- 신뢰도 60%+ 신호의 매수/매도 합산 수익률 최대화
- 매수/매도 신호 각각 10개 미만 시 강한 패널티 (×0.3)
- 매수/매도 중 하나라도 수익 0 이하 시 패널티 (×0.5)
- **seed=42 고정** → 재현 가능한 결과 보장

---

## 출력 예시

```
====================================================
  AAPL  |  2025-03-17 14:30  |  $215.42
----------------------------------------------------
  다음 5분봉 예측:  📈 BULLISH  (강한 상승)
  신뢰도:          68.3%
----------------------------------------------------
  예상 변동폭:     +0.231%
  예상 종가:       $215.92

  시나리오별 확률:
    📈 상승  ████████████░░░░░░░░░░░░░░░░░░░  68.3%
    📉 하락  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░  18.1%
    ➡️  횡보  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░  13.6%
====================================================
```

---

## 의존성

```bash
pip install yfinance pandas numpy scikit-learn xgboost optuna torch --break-system-packages
```
