"""
XGBoost 범용 모델 학습 + 백테스트
====================================
사용법:
  pip install xgboost scikit-learn pandas numpy matplotlib --break-system-packages

  # combined_5m_features.csv로 학습
  python train_model.py

  # 특정 파일 지정
  python train_model.py --data output/combined_5m_features.csv

출력:
  models/chart_model.json      ← 저장된 XGBoost 모델
  models/feature_list.txt      ← 피처 목록 (predict.py에서 사용)
  models/backtest_report.txt   ← 백테스트 결과
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("xgboost 미설치. 설치 후 재실행: pip install xgboost --break-system-packages")
    HAS_XGB = False


# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 & 전처리
# ─────────────────────────────────────────────────────────

def load_data(path: str):
    print(f"\n데이터 로드: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  전체 행: {len(df):,}개")

    if 'ticker' in df.columns:
        print(f"  종목: {df['ticker'].unique().tolist()}")
        print(f"  종목별 행수:")
        for t, cnt in df['ticker'].value_counts().items():
            print(f"    {t}: {cnt:,}행")

    return df


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """피처/라벨 분리 + 결측값 처리"""
    X = df[feature_cols].copy()
    y = df['label'].copy()  # 1=BULLISH, -1=BEARISH, 0=NEUTRAL

    # XGBoost용 라벨 변환 (0, 1, 2)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # -1→0, 0→1, 1→2

    # 무한값 처리
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    print(f"\n  피처 수: {len(feature_cols)}개")
    print(f"  샘플 수: {len(X):,}개")
    print(f"\n  라벨 분포:")
    label_names = {0: 'BEARISH(-1)', 1: 'NEUTRAL(0)', 2: 'BULLISH(1)'}
    for enc_val, cnt in zip(*np.unique(y_enc, return_counts=True)):
        pct = cnt / len(y_enc) * 100
        bar = '█' * int(pct / 2)
        print(f"    {label_names[enc_val]:12s}: {cnt:5,}개 ({pct:.1f}%) {bar}")

    return X, y_enc, le


def get_feature_cols(df: pd.DataFrame) -> list:
    """학습에 쓸 피처 컬럼만 추출"""
    exclude = {
        'open', 'high', 'low', 'close', 'volume',
        'label', 'label_name', 'future_return_pct', 'ticker',
        'ma9', 'ma20', 'ma50', 'ma200',
        'bb_upper', 'bb_lower', 'macd_signal', 'stoch_d',
        'vol_ma12', 'vol_ma50', 'atr14', 'obv', 'vwap',
        'high_1h', 'low_1h', 'high_4h', 'low_4h', 'high_1d', 'low_1d',
    }
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


# ─────────────────────────────────────────────────────────
# 2. 시계열 교차검증 학습
# ─────────────────────────────────────────────────────────

def train_with_cv(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
    """
    TimeSeriesSplit: 미래 데이터가 학습에 새어들어가지 않도록
    일반 KFold 대신 시계열 전용 교차검증 사용
    """
    print(f"\n{'─'*52}")
    print(f"  시계열 교차검증 ({n_splits}-fold)")
    print(f"{'─'*52}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    params = {
        'objective':        'multi:softprob',
        'num_class':        3,              # BEARISH / NEUTRAL / BULLISH
        'n_estimators':     300,
        'max_depth':        5,
        'learning_rate':    0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma':            0.1,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'random_state':     42,
        'n_jobs':           -1,
        'verbosity':        0,
        'eval_metric':      'mlogloss',
    }

    X_arr = X.values
    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx],     y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        fold_scores.append(acc)

        if acc > best_score:
            best_score = acc
            best_model = model

        print(f"  Fold {fold+1}/{n_splits}  정확도: {acc:.4f}  (train {len(train_idx):,} / val {len(val_idx):,})")

    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)
    print(f"\n  CV 평균 정확도: {mean_acc:.4f} ± {std_acc:.4f}")

    return best_model, fold_scores


# ─────────────────────────────────────────────────────────
# 3. 최종 모델 전체 데이터로 재학습
# ─────────────────────────────────────────────────────────

def train_final(X: pd.DataFrame, y: np.ndarray):
    """CV 완료 후 전체 데이터로 최종 모델 학습"""
    print(f"\n  전체 데이터로 최종 모델 학습 중...")

    params = {
        'objective':        'multi:softprob',
        'num_class':        3,
        'n_estimators':     300,
        'max_depth':        5,
        'learning_rate':    0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma':            0.1,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'random_state':     42,
        'n_jobs':           -1,
        'verbosity':        0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X.values, y, verbose=False)
    print(f"  최종 모델 학습 완료")
    return model


# ─────────────────────────────────────────────────────────
# 4. 백테스트
# ─────────────────────────────────────────────────────────

def backtest(model, X: pd.DataFrame, df_orig: pd.DataFrame, le: LabelEncoder, feature_cols: list):
    """
    마지막 20%를 hold-out 테스트셋으로 백테스트
    - 신호가 나왔을 때 실제로 수익이 났는지 검증
    - confidence threshold 적용 (불확실한 예측 제외)
    """
    print(f"\n{'─'*52}")
    print(f"  백테스트 (Hold-out 마지막 20%)")
    print(f"{'─'*52}")

    n = len(X)
    split = int(n * 0.8)

    X_test   = X.iloc[split:]
    df_test  = df_orig.iloc[split:].copy()

    X_test_arr = X_test.values.copy()
    X_test_arr = np.where(np.isinf(X_test_arr), np.nan, X_test_arr)
    X_test_filled = pd.DataFrame(X_test_arr, columns=feature_cols)
    X_test_filled.fillna(X_test_filled.median(), inplace=True)

    # 확률 예측
    proba = model.predict_proba(X_test_filled.values)
    # 클래스 순서: 0=BEARISH(-1), 1=NEUTRAL(0), 2=BULLISH(1)
    prob_bear   = proba[:, 0]
    prob_neut   = proba[:, 1]
    prob_bull   = proba[:, 2]
    pred_labels = model.predict(X_test_filled.values)

    # 원래 라벨로 디코딩
    pred_orig = le.inverse_transform(pred_labels)  # -1, 0, 1
    true_orig = df_test['label'].values

    # 전체 정확도
    acc = (pred_orig == true_orig).mean()
    print(f"\n  전체 정확도: {acc:.4f} ({acc*100:.1f}%)")

    # 분류 리포트
    print(f"\n  분류 리포트:")
    report = classification_report(
        true_orig, pred_orig,
        target_names=['BEARISH', 'NEUTRAL', 'BULLISH'],
        zero_division=0
    )
    for line in report.split('\n'):
        print(f"    {line}")

    # Confidence Threshold 백테스트
    print(f"\n  신뢰도별 성과 (BULLISH 신호만):")
    print(f"  {'임계값':8s} {'신호수':6s} {'정확도':8s} {'평균수익률':10s}")
    print(f"  {'─'*40}")

    results = []
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        bull_mask = (prob_bull >= thresh)
        if bull_mask.sum() == 0:
            continue
        correct = (pred_orig[bull_mask] == true_orig[bull_mask]).mean()
        avg_ret = df_test['future_return_pct'].values[bull_mask].mean()
        n_sig   = bull_mask.sum()
        results.append({'thresh': thresh, 'n': n_sig, 'acc': correct, 'avg_ret': avg_ret})
        print(f"  {thresh:.0%}      {n_sig:5d}   {correct:.4f}    {avg_ret:+.3f}%")

    print(f"\n  BEARISH 신호 성과:")
    print(f"  {'임계값':8s} {'신호수':6s} {'정확도':8s} {'평균수익률':10s}")
    print(f"  {'─'*40}")
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        bear_mask = (prob_bear >= thresh)
        if bear_mask.sum() == 0:
            continue
        correct = (pred_orig[bear_mask] == true_orig[bear_mask]).mean()
        avg_ret = df_test['future_return_pct'].values[bear_mask].mean()
        n_sig   = bear_mask.sum()
        print(f"  {thresh:.0%}      {n_sig:5d}   {correct:.4f}    {avg_ret:+.3f}%")

    # 피처 중요도 Top 15
    print(f"\n  피처 중요도 Top 15:")
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    max_imp = feat_imp[0][1]
    for feat, imp in feat_imp:
        bar = '█' * int(imp / max_imp * 25)
        print(f"    {feat:25s} {imp:.4f}  {bar}")

    return results


# ─────────────────────────────────────────────────────────
# 5. 모델 저장
# ─────────────────────────────────────────────────────────

def save_model(model, feature_cols: list, le: LabelEncoder, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    # XGBoost 모델 저장
    model_path = os.path.join(output_dir, 'chart_model.json')
    model.save_model(model_path)

    # 피처 목록 저장
    feat_path = os.path.join(output_dir, 'feature_list.txt')
    with open(feat_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')

    # 라벨 인코더 클래스 저장
    le_path = os.path.join(output_dir, 'label_classes.json')
    with open(le_path, 'w') as f:
        json.dump(le.classes_.tolist(), f)

    print(f"\n  모델 저장: {model_path}")
    print(f"  피처 목록: {feat_path}")
    print(f"  라벨 클래스: {le_path}")
    print(f"\n  → 다음 단계: python predict.py --ticker AAPL")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XGBoost 5분봉 모델 학습')
    parser.add_argument('--data',   default='output/combined_5m_features.csv', help='학습 데이터 경로')
    parser.add_argument('--output', default='models', help='모델 저장 디렉토리')
    parser.add_argument('--cv',     type=int, default=5, help='교차검증 fold 수')
    args = parser.parse_args()

    if not HAS_XGB:
        return

    print(f"\nXGBoost 5분봉 모델 학습")
    print(f"{'='*52}")

    # 데이터 로드
    df = load_data(args.data)
    feature_cols = get_feature_cols(df)
    X, y_enc, le = prepare_features(df, feature_cols)

    # 교차검증
    best_cv_model, cv_scores = train_with_cv(X, y_enc, n_splits=args.cv)

    # 전체 데이터로 최종 학습
    final_model = train_final(X, y_enc)

    # 백테스트
    backtest(final_model, X, df, le, feature_cols)

    # 저장
    save_model(final_model, feature_cols, le, output_dir=args.output)

    print(f"\n{'='*52}")
    print(f"  학습 완료!")
    print(f"  CV 평균 정확도: {np.mean(cv_scores):.4f}")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    main()
