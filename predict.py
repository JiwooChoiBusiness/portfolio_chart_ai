"""
티커 입력 → 저장된 모델로 즉시 예측
======================================
사용법:
  python predict.py --ticker AAPL
  python predict.py --ticker NVDA --bars 3   # 최근 3봉 예측

출력:
  현재 5분봉 기준 다음 봉 예측 결과
  (상승/하락/횡보 확률 + Claude 앱에 넘길 JSON)
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("pip install xgboost --break-system-packages")
    HAS_XGB = False


# ─────────────────────────────────────────────────────────
# 피처 생성 (data_pipeline_5m.py와 동일한 로직)
# ─────────────────────────────────────────────────────────

def fetch_latest(ticker: str, bars: int = 200) -> pd.DataFrame:
    """예측에 필요한 최신 데이터 수집 (이동평균 계산용 충분한 봉 수)"""
    print(f"  [{ticker}] 최신 5분봉 수집 중...")
    df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"데이터를 가져오지 못했습니다: {ticker}")

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York')
    df = df.iloc[df.index.indexer_between_time('09:30', '15:55')]
    df.dropna(inplace=True)

    print(f"  [{ticker}] {len(df)}봉 수집 (최신: {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """data_pipeline_5m.py의 피처 생성 로직 동일하게 적용"""
    from data_pipeline_5m import (
        add_candle_features, add_pattern_features, add_ma_features,
        add_vwap_features, add_session_features, add_momentum_features,
        add_volume_features, add_volatility_features,
        add_support_resistance_features, add_lag_features
    )

    df = add_candle_features(df)
    df = add_pattern_features(df)
    df = add_ma_features(df)
    df = add_vwap_features(df)
    df = add_session_features(df)
    df = add_momentum_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_support_resistance_features(df)
    df = add_lag_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ─────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────

def load_model(model_dir: str = "models"):
    model_path = os.path.join(model_dir, 'chart_model.json')
    feat_path  = os.path.join(model_dir, 'feature_list.txt')
    le_path    = os.path.join(model_dir, 'label_classes.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}\n먼저 python train_model.py 실행")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    with open(le_path) as f:
        label_classes = json.load(f)  # [-1, 0, 1]

    print(f"  모델 로드 완료 (피처 {len(feature_cols)}개)")
    return model, feature_cols, label_classes


# ─────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────

def predict(model, feature_cols: list, label_classes: list, df: pd.DataFrame, n_latest: int = 1):
    """
    최근 n_latest봉에 대한 예측 수행
    반환: 각 봉의 예측 결과 리스트
    """
    # 모델이 학습한 피처 컬럼만 선택 (없는 컬럼은 0으로 채움)
    X = pd.DataFrame(index=df.index[-n_latest:])
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col].iloc[-n_latest:].values
        else:
            X[col] = 0.0

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    proba  = model.predict_proba(X.values)
    preds  = model.predict(X.values)

    # label_classes = [-1, 0, 1] → 인덱스 0=BEARISH, 1=NEUTRAL, 2=BULLISH
    results = []
    for i in range(len(X)):
        lc = label_classes
        # 클래스별 확률 매핑
        prob_map = {int(lc[j]): float(proba[i][j]) for j in range(len(lc))}
        prob_bull = prob_map.get(1,  0.0)
        prob_neut = prob_map.get(0,  0.0)
        prob_bear = prob_map.get(-1, 0.0)

        pred_label = int(preds[i]) if preds[i] in lc else 0
        # preds는 인코딩된 값(0,1,2)이므로 원래 라벨로 변환
        pred_orig  = int(label_classes[int(preds[i])])

        # 신호 강도 판정
        max_prob = max(prob_bull, prob_bear, prob_neut)
        if pred_orig == 1:
            direction = 'BULLISH'
            if prob_bull >= 0.65:   strength = '강한 상승'
            elif prob_bull >= 0.50: strength = '약한 상승'
            else:                   strength = '불확실'
        elif pred_orig == -1:
            direction = 'BEARISH'
            if prob_bear >= 0.65:   strength = '강한 하락'
            elif prob_bear >= 0.50: strength = '약한 하락'
            else:                   strength = '불확실'
        else:
            direction = 'NEUTRAL'
            strength  = '불확실'

        row = df.iloc[-(n_latest - i)]
        results.append({
            'timestamp':    X.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(X.index[i], 'strftime') else str(X.index[i]),
            'close':        float(row['close']),
            'prediction':   direction,
            'strength':     strength,
            'probability': {
                'bullish': round(prob_bull * 100, 1),
                'bearish': round(prob_bear * 100, 1),
                'neutral': round(prob_neut * 100, 1),
            },
            'confidence':   round(max_prob * 100, 1),
        })

    return results


# ─────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────

def print_result(r: dict, ticker: str):
    icons = {'BULLISH': '📈', 'BEARISH': '📉', 'NEUTRAL': '➡️'}
    colors = {'BULLISH': '\033[92m', 'BEARISH': '\033[91m', 'NEUTRAL': '\033[93m'}
    reset  = '\033[0m'

    pred   = r['prediction']
    color  = colors.get(pred, '')
    icon   = icons.get(pred, '')

    print(f"\n{'='*52}")
    print(f"  {ticker}  |  {r['timestamp']}  |  ${r['close']:.2f}")
    print(f"{'─'*52}")
    print(f"  다음 5분봉 예측:  {color}{icon} {r['prediction']}  ({r['strength']}){reset}")
    print(f"  신뢰도:          {r['confidence']:.1f}%")
    print(f"\n  시나리오별 확률:")

    for label, pct in [('📈 상승', r['probability']['bullish']),
                        ('📉 하락', r['probability']['bearish']),
                        ('➡️  횡보', r['probability']['neutral'])]:
        bar_len = int(pct / 3)
        bar = '█' * bar_len + '░' * (33 - bar_len)
        print(f"    {label}  {bar}  {pct:.1f}%")

    print(f"{'='*52}")


def export_for_app(results: list, ticker: str, output_dir: str = "output"):
    """Claude 앱(chart-predictor-v2.html)에서 바로 쓸 수 있는 JSON 포맷으로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    latest = results[-1]
    app_payload = {
        "ticker":     ticker,
        "timestamp":  latest['timestamp'],
        "ml_result": {
            "prediction":  latest['prediction'],
            "strength":    latest['strength'],
            "probability": latest['probability'],
            "confidence":  latest['confidence'],
            "source":      "XGBoost 5분봉 범용 모델"
        }
    }

    path = os.path.join(output_dir, f"{ticker}_ml_signal.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(app_payload, f, ensure_ascii=False, indent=2)

    print(f"\n  앱 연동용 JSON 저장: {path}")
    print(f"  (chart-predictor-v2.html의 ML 신호로 활용 가능)")
    return path


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='5분봉 XGBoost 예측')
    parser.add_argument('--ticker',    required=True,  help='예측할 종목 티커')
    parser.add_argument('--bars',      type=int, default=1, help='최근 N봉 예측 (기본 1)')
    parser.add_argument('--models',    default='models', help='모델 디렉토리')
    parser.add_argument('--output',    default='output', help='결과 저장 디렉토리')
    args = parser.parse_args()

    if not HAS_XGB:
        return

    ticker = args.ticker.upper()
    print(f"\n5분봉 예측: {ticker}")
    print(f"{'='*52}")

    # 모델 로드
    model, feature_cols, label_classes = load_model(args.models)

    # 최신 데이터 수집 + 피처 생성
    df_raw = fetch_latest(ticker)
    df     = build_features(df_raw)

    if len(df) < args.bars:
        print(f"  데이터 부족: {len(df)}봉 (요청 {args.bars}봉)")
        return

    # 예측
    results = predict(model, feature_cols, label_classes, df, n_latest=args.bars)

    # 출력
    for r in results:
        print_result(r, ticker)

    # 앱 연동용 JSON 저장
    export_for_app(results, ticker, output_dir=args.output)


if __name__ == '__main__':
    main()
