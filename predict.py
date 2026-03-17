"""
티커 입력 → XGBoost + CNN 앙상블 예측 v3
=========================================
변경사항 (v2 → v3):
  - CNN 모델 추가 (models/cnn_model.pt)
  - XGBoost + CNN 앙상블 예측
  - CNN 없을 시 XGBoost 단독으로 자동 fallback

사용법:
  python predict.py --ticker AAPL
  python predict.py --ticker NVDA --bars 3
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

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False


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


def build_features(df: pd.DataFrame, ticker: str = '') -> pd.DataFrame:
    """data_pipeline_5m.py의 피처 생성 로직 동일하게 적용"""
    from data_pipeline_5m import (
        add_candle_features, add_pattern_features, add_ma_features,
        add_vwap_features, add_session_features, add_momentum_features,
        add_volume_features, add_volatility_features,
        add_support_resistance_features, add_lag_features,
        add_vix_features,  fetch_vix_features,
        add_gold_features, fetch_gold_features,
        add_oil_features,  fetch_oil_features,
        add_dxy_features,  fetch_dxy_features,
        add_sector_features, add_interaction_features,
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
    df = add_vix_features(df,  fetch_vix_features())
    df = add_gold_features(df, fetch_gold_features())
    df = add_oil_features(df,  fetch_oil_features())
    df = add_dxy_features(df,  fetch_dxy_features())
    df = add_sector_features(df, ticker)
    df = add_interaction_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ─────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────

def load_model(model_dir: str = "models"):
    model_path = os.path.join(model_dir, 'chart_model.json')
    reg_path   = os.path.join(model_dir, 'chart_model_reg.json')
    feat_path  = os.path.join(model_dir, 'feature_list.txt')
    le_path    = os.path.join(model_dir, 'label_classes.json')
    params_path= os.path.join(model_dir, 'best_params.json')
    cnn_path   = os.path.join(model_dir, 'cnn_model.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}\n먼저 python train_model.py 실행")

    # XGBoost 분류 모델 로드
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    # XGBoost 회귀 모델 로드 (있을 때만)
    reg_model = None
    if os.path.exists(reg_path):
        reg_model = xgb.XGBRegressor()
        reg_model.load_model(reg_path)
        print(f"  회귀 모델 로드 완료")
    else:
        print(f"  회귀 모델 없음 (종가 예측 불가)")

    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    with open(le_path) as f:
        label_classes = json.load(f)

    # 앙상블 가중치 로드
    xgb_w, cnn_w = 1.0, 0.0
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        ew = params.get('ensemble_weight', {})
        xgb_w = ew.get('xgb', 1.0)
        cnn_w  = ew.get('cnn', 0.0)

    # CNN 로드 (있을 때만)
    cnn_model = None
    if HAS_TORCH and os.path.exists(cnn_path) and cnn_w > 0:
        from train_model import CNN1D
        ckpt      = torch.load(cnn_path, map_location=DEVICE)
        cnn_model = CNN1D(
            n_features = ckpt['n_features'],
            seq_len    = ckpt['seq_len'],
            n_classes  = ckpt['n_classes'],
        ).to(DEVICE)
        cnn_model.load_state_dict(ckpt['state_dict'])
        cnn_model.eval()
        print(f"  CNN 모델 로드 완료 (seq_len={ckpt['seq_len']})")

    mode = "XGBoost + CNN 앙상블" if cnn_model else "XGBoost 단독"
    print(f"  모델 로드 완료 | {mode} | 피처 {len(feature_cols)}개")
    print(f"  앙상블 가중치: XGB {xgb_w:.1f} / CNN {cnn_w:.1f}")

    return xgb_model, reg_model, cnn_model, feature_cols, label_classes, xgb_w, cnn_w


# ─────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────

def predict(xgb_model, reg_model, cnn_model, feature_cols: list, label_classes: list,
            df: pd.DataFrame, n_latest: int = 1,
            xgb_w: float = 1.0, cnn_w: float = 0.0, seq_len: int = 20):
    """
    XGBoost + CNN 앙상블 분류 예측 + 회귀 예측 (변동폭/종가 범위)
    """
    # ── XGBoost 예측 ─────────────────────────────────────
    X = pd.DataFrame(index=df.index[-n_latest:])
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col].iloc[-n_latest:].values
        else:
            X[col] = 0.0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    xgb_proba = xgb_model.predict_proba(X.values)  # (n, 3)

    # ── 회귀 예측 (변동폭) ────────────────────────────────
    reg_preds = None
    if reg_model is not None:
        reg_preds = reg_model.predict(X.values)  # (n,) 변동폭 %

    # ── CNN 예측 (있을 때만) ──────────────────────────────
    cnn_proba = None
    if cnn_model is not None and HAS_TORCH and len(df) >= seq_len + n_latest:
        feat_arr = df[feature_cols].values if all(c in df.columns for c in feature_cols) \
                   else np.zeros((len(df), len(feature_cols)))
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        seqs = []
        for i in range(n_latest):
            idx = len(df) - n_latest + i
            if idx >= seq_len:
                seqs.append(feat_arr[idx - seq_len:idx])
            else:
                seqs.append(np.zeros((seq_len, len(feature_cols)), dtype=np.float32))

        seq_tensor = torch.tensor(np.array(seqs)).to(DEVICE)
        with torch.no_grad():
            logits    = cnn_model(seq_tensor)
            cnn_proba = torch.softmax(logits, dim=1).cpu().numpy()

    # ── 앙상블 가중 평균 ──────────────────────────────────
    if cnn_proba is not None:
        total = xgb_w + cnn_w
        proba = (xgb_proba * xgb_w + cnn_proba * cnn_w) / total
        source = f"XGBoost({xgb_w:.1f}) + CNN({cnn_w:.1f}) 앙상블"
    else:
        proba  = xgb_proba
        source = "XGBoost 단독"

    preds = proba.argmax(axis=1)

    # ── 결과 생성 ─────────────────────────────────────────
    results = []
    for i in range(n_latest):
        prob_map  = {int(label_classes[j]): float(proba[i][j]) for j in range(len(label_classes))}
        prob_bull = prob_map.get(1,  0.0)
        prob_neut = prob_map.get(0,  0.0)
        prob_bear = prob_map.get(-1, 0.0)
        pred_orig = int(label_classes[int(preds[i])])
        max_prob  = max(prob_bull, prob_bear, prob_neut)

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

        row        = df.iloc[-(n_latest - i)]
        curr_close = float(row['close'])

        # 회귀 예측 결과 (변동폭 → 종가 환산)
        pred_change  = float(reg_preds[i]) if reg_preds is not None else None
        pred_close   = round(curr_close * (1 + pred_change / 100), 2) if pred_change is not None else None

        results.append({
            'timestamp':    X.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(X.index[i], 'strftime') else str(X.index[i]),
            'close':        curr_close,
            'prediction':   direction,
            'strength':     strength,
            'probability': {
                'bullish': round(prob_bull * 100, 1),
                'bearish': round(prob_bear * 100, 1),
                'neutral': round(prob_neut * 100, 1),
            },
            'confidence':   round(max_prob * 100, 1),
            'source':       source,
            'pred_change':  round(pred_change, 3) if pred_change is not None else None,
            'pred_close':   pred_close,
        })

    return results
    """
    XGBoost + CNN 앙상블 예측
    CNN 없을 시 XGBoost 단독 예측
    """
    # ── XGBoost 예측 ─────────────────────────────────────
    X = pd.DataFrame(index=df.index[-n_latest:])
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col].iloc[-n_latest:].values
        else:
            X[col] = 0.0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    xgb_proba = xgb_model.predict_proba(X.values)  # (n, 3)

    # ── CNN 예측 (있을 때만) ──────────────────────────────
    cnn_proba = None
    if cnn_model is not None and HAS_TORCH and len(df) >= seq_len + n_latest:
        feat_arr = df[feature_cols].values if all(c in df.columns for c in feature_cols) \
                   else np.zeros((len(df), len(feature_cols)))
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        seqs = []
        for i in range(n_latest):
            idx = len(df) - n_latest + i
            if idx >= seq_len:
                seqs.append(feat_arr[idx - seq_len:idx])
            else:
                seqs.append(np.zeros((seq_len, len(feature_cols)), dtype=np.float32))

        seq_tensor = torch.tensor(np.array(seqs)).to(DEVICE)
        with torch.no_grad():
            logits    = cnn_model(seq_tensor)
            cnn_proba = torch.softmax(logits, dim=1).cpu().numpy()

    # ── 앙상블 가중 평균 ──────────────────────────────────
    if cnn_proba is not None:
        total = xgb_w + cnn_w
        proba = (xgb_proba * xgb_w + cnn_proba * cnn_w) / total
        source = f"XGBoost({xgb_w:.1f}) + CNN({cnn_w:.1f}) 앙상블"
    else:
        proba  = xgb_proba
        source = "XGBoost 단독"

    preds = proba.argmax(axis=1)

    # ── 결과 생성 ─────────────────────────────────────────
    results = []
    for i in range(n_latest):
        prob_map  = {int(label_classes[j]): float(proba[i][j]) for j in range(len(label_classes))}
        prob_bull = prob_map.get(1,  0.0)
        prob_neut = prob_map.get(0,  0.0)
        prob_bear = prob_map.get(-1, 0.0)
        pred_orig = int(label_classes[int(preds[i])])
        max_prob  = max(prob_bull, prob_bear, prob_neut)

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
            'source':       source,
        })

    return results


# ─────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────

def print_result(r: dict, ticker: str):
    icons  = {'BULLISH': '📈', 'BEARISH': '📉', 'NEUTRAL': '➡️'}
    colors = {'BULLISH': '\033[92m', 'BEARISH': '\033[91m', 'NEUTRAL': '\033[93m'}
    reset  = '\033[0m'

    pred  = r['prediction']
    color = colors.get(pred, '')
    icon  = icons.get(pred, '')

    print(f"\n{'='*52}")
    print(f"  {ticker}  |  {r['timestamp']}  |  ${r['close']:.2f}")
    print(f"{'─'*52}")
    print(f"  다음 5분봉 예측:  {color}{icon} {r['prediction']}  ({r['strength']}){reset}")
    print(f"  신뢰도:          {r['confidence']:.1f}%")

    # 회귀 예측 출력
    if r.get('pred_change') is not None:
        print(f"{'─'*52}")
        print(f"  예상 변동폭:     {r['pred_change']:+.3f}%")
        print(f"  예상 종가:       ${r['pred_close']:.2f}")

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
    parser = argparse.ArgumentParser(description='5분봉 XGBoost + CNN 앙상블 예측 v3')
    parser.add_argument('--ticker',    required=True,  help='예측할 종목 티커')
    parser.add_argument('--bars',      type=int, default=1, help='최근 N봉 예측 (기본 1)')
    parser.add_argument('--models',    default='models', help='모델 디렉토리')
    parser.add_argument('--output',    default='output', help='결과 저장 디렉토리')
    parser.add_argument('--seq-len',   type=int, default=20, help='CNN 시퀀스 길이 (기본 20)')
    args = parser.parse_args()

    if not HAS_XGB:
        return

    ticker = args.ticker.upper()
    print(f"\n5분봉 앙상블 예측 v3: {ticker}")
    print(f"{'='*52}")

    # 모델 로드
    xgb_model, reg_model, cnn_model, feature_cols, label_classes, xgb_w, cnn_w = load_model(args.models)

    # 최신 데이터 수집 + 피처 생성
    df_raw = fetch_latest(ticker)
    df     = build_features(df_raw, ticker=ticker)

    if len(df) < args.bars:
        print(f"  데이터 부족: {len(df)}봉 (요청 {args.bars}봉)")
        return

    # 앙상블 예측
    results = predict(
        xgb_model, reg_model, cnn_model, feature_cols, label_classes, df,
        n_latest = args.bars,
        xgb_w    = xgb_w,
        cnn_w    = cnn_w,
        seq_len  = args.seq_len,
    )

    # 출력
    for r in results:
        print_result(r, ticker)

    # 앱 연동용 JSON 저장
    export_for_app(results, ticker, output_dir=args.output)


if __name__ == '__main__':
    main()
