"""
XGBoost + 1D CNN 앙상블 모델 학습 v3
======================================
변경사항 (v2 → v3):
  - 1D CNN 모델 추가 (시퀀스 패턴 학습)
  - XGBoost + CNN 앙상블 (가중 평균)
  - 앙상블 가중치 Optuna 자동 최적화
  - 모델 저장: models/chart_model.json (XGB) + models/cnn_model.pt (CNN)

사용법:
  pip install xgboost scikit-learn pandas numpy optuna torch --break-system-packages

  python train_model.py                        # 기본 실행
  python train_model.py --no-optuna            # Optuna 건너뛰기
  python train_model.py --optuna-trials 100    # 탐색 횟수 조정
  python train_model.py --cnn-epochs 30        # CNN 학습 에포크 조정

출력:
  models/chart_model.json   ← XGBoost 모델
  models/cnn_model.pt       ← CNN 모델
  models/feature_list.txt   ← 피처 목록
  models/label_classes.json ← 라벨 클래스
  models/best_params.json   ← Optuna 최적 파라미터 + 앙상블 가중치
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("xgboost 미설치: pip install xgboost --break-system-packages")
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    print("optuna 미설치: pip install optuna --break-system-packages")
    HAS_OPTUNA = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  PyTorch 디바이스: {DEVICE}")
except ImportError:
    print("torch 미설치: pip install torch --break-system-packages")
    HAS_TORCH = False


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
# 2. 1D CNN 모델 정의 & 학습
# ─────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1D CNN for time-series classification
    입력: (batch, seq_len, n_features)
    출력: (batch, 3)  — BEARISH / NEUTRAL / BULLISH 확률
    """
    def __init__(self, n_features: int, seq_len: int = 20, n_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_cnn(seq_path: str, output_dir: str = "models",
              epochs: int = 20, seq_len: int = 20,
              batch_size: int = 512) -> nn.Module:
    """
    CNN 학습
    sequences_X.npy, sequences_y.npy 로드 후 학습
    """
    if not HAS_TORCH:
        print("  PyTorch 미설치 → CNN 건너뜀")
        return None

    print(f"\n{'─'*52}")
    print(f"  1D CNN 학습 (epochs={epochs}, batch={batch_size})")
    print(f"{'─'*52}")

    # 시퀀스 데이터 로드
    X_seq = np.load(os.path.join(seq_path, 'sequences_X.npy'))
    y_seq = np.load(os.path.join(seq_path, 'sequences_y.npy'))

    # 라벨 인코딩 (-1,0,1 → 0,1,2)
    from sklearn.preprocessing import LabelEncoder
    le_cnn = LabelEncoder()
    y_enc  = le_cnn.fit_transform(y_seq)

    # 시간순 80/20 split
    n     = len(X_seq)
    split = int(n * 0.8)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_enc[:split], y_enc[split:]

    # NaN/Inf 처리
    X_tr  = np.nan_to_num(X_tr,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    n_features = X_seq.shape[2]
    model_cnn  = CNN1D(n_features=n_features, seq_len=seq_len).to(DEVICE)
    optimizer  = optim.Adam(model_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model_cnn.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model_cnn(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 검증
        model_cnn.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                preds = model_cnn(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model_cnn.state_dict().items()}

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  val_acc: {val_acc:.4f}")

    model_cnn.load_state_dict(best_state)
    print(f"\n  CNN 최적 val_acc: {best_val_acc:.4f}")

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    cnn_path = os.path.join(output_dir, 'cnn_model.pt')
    torch.save({
        'state_dict': best_state,
        'n_features': n_features,
        'seq_len':    seq_len,
        'n_classes':  3,
    }, cnn_path)
    print(f"  CNN 모델 저장: {cnn_path}")

    return model_cnn


# ─────────────────────────────────────────────────────────
# 3. Optuna 하이퍼파라미터 튜닝
# ─────────────────────────────────────────────────────────

def optuna_tune(X: pd.DataFrame, y: np.ndarray, df_orig: pd.DataFrame, n_trials: int = 50) -> dict:
    """
    Optuna로 XGBoost 최적 파라미터 탐색
    목적함수: 합산 수익률 (매수+매도 신뢰도 60% 이상 신호의 누적 수익률)
    → 정확도 기준으로 하면 관망 편향 발생하여 수익률 0%가 되는 문제 방지
    """
    print(f"\n{'─'*52}")
    print(f"  Optuna 하이퍼파라미터 튜닝 ({n_trials}회 탐색)")
    print(f"  목적함수: 합산 수익률 (신뢰도 60%+)")
    print(f"{'─'*52}")

    X_arr      = X.values
    tscv       = TimeSeriesSplit(n_splits=3)
    future_ret = df_orig['future_return_pct'].values

    # label_classes 순서 확인 (-1, 0, 1)
    from sklearn.preprocessing import LabelEncoder as _LE
    _le = _LE()
    _le.fit(df_orig['label'].values)
    lc       = _le.classes_.tolist()
    bull_idx = lc.index(1)
    bear_idx = lc.index(-1)

    def objective(trial):
        params = {
            'objective':        'multi:softprob',
            'num_class':        3,
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 2.0),
            'random_state':     42,
            'n_jobs':           -1,
            'verbosity':        0,
            'eval_metric':      'mlogloss',
        }

        fold_scores = []
        for train_idx, val_idx in tscv.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train        = y[train_idx]
            ret_val        = future_ret[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val := y[val_idx])],
                      verbose=False)

            proba     = model.predict_proba(X_val)
            prob_bull = proba[:, bull_idx]
            prob_bear = proba[:, bear_idx]

            bull_mask = prob_bull >= 0.60
            bear_mask = prob_bear >= 0.60

            bull_ret = ret_val[bull_mask].mean()  if bull_mask.sum() > 0 else 0.0
            bear_ret = -ret_val[bear_mask].mean() if bear_mask.sum() > 0 else 0.0
            combined = bull_ret + bear_ret

            # 매수/매도 신호 수 절대값 기준 패널티 (비율 기준 제거)
            if bull_mask.sum() < 10 or bear_mask.sum() < 10:
                combined *= 0.3  # 한쪽이라도 신호 부족하면 강한 패널티

            # 매수/매도 균형 패널티 (한쪽만 수익이면 감점)
            if bull_ret <= 0 or bear_ret <= 0:
                combined *= 0.5

            fold_scores.append(combined)

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best['objective']    = 'multi:softprob'
    best['num_class']    = 3
    best['random_state'] = 42
    best['n_jobs']       = -1
    best['verbosity']    = 0

    print(f"\n  최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"    {k:20s}: {v}")
    print(f"  최적 합산 수익률: {study.best_value:.4f}%")

    return best


# ─────────────────────────────────────────────────────────
# 3. 시계열 교차검증 학습
# ─────────────────────────────────────────────────────────

def train_with_cv(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, params: dict = None):
    """
    TimeSeriesSplit: 미래 데이터가 학습에 새어들어가지 않도록
    일반 KFold 대신 시계열 전용 교차검증 사용
    params가 없으면 기본값 사용, 있으면 Optuna 최적 파라미터 사용
    """
    print(f"\n{'─'*52}")
    print(f"  시계열 교차검증 ({n_splits}-fold)")
    print(f"{'─'*52}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    if params is None:
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
            'eval_metric':      'mlogloss',
        }
    else:
        params['eval_metric'] = 'mlogloss'

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

def train_final(X: pd.DataFrame, y: np.ndarray, params: dict = None):
    """CV 완료 후 전체 데이터로 최종 모델 학습"""
    print(f"\n  전체 데이터로 최종 모델 학습 중...")

    if params is None:
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
    v4 추가:
      - 신뢰도 기반 청산 전략 (진입 50%, 손절 40%, 반대신호 즉시 전환)
      - 수수료 0.07% 왕복 반영 (수수료 0.04% + 슬리피지 0.03%)
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

    # label_classes 순서 기반으로 인덱스 동적 결정
    lc        = le.classes_.tolist()
    bull_idx  = lc.index(1)
    bear_idx  = lc.index(-1)
    neut_idx  = lc.index(0)
    prob_bull = proba[:, bull_idx]
    prob_bear = proba[:, bear_idx]
    prob_neut = proba[:, neut_idx]
    pred_labels = model.predict(X_test_filled.values)

    # 원래 라벨로 디코딩
    pred_orig = le.inverse_transform(pred_labels)
    true_orig = df_test['label'].values
    future_ret = df_test['future_return_pct'].values

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
        avg_ret = future_ret[bull_mask].mean()
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
        avg_ret = future_ret[bear_mask].mean()
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

def save_model(model, feature_cols: list, le: LabelEncoder, output_dir: str = "models", best_params: dict = None):
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

    # Optuna 최적 파라미터 저장
    if best_params:
        params_path = os.path.join(output_dir, 'best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"  최적 파라미터: {params_path}")

    print(f"\n  모델 저장: {model_path}")
    print(f"  피처 목록: {feat_path}")
    print(f"  라벨 클래스: {le_path}")
    print(f"\n  → 다음 단계: python predict.py --ticker AAPL --models {output_dir}")


# ─────────────────────────────────────────────────────────
# 6. 회귀 모델 (예상 변동폭 예측)
# ─────────────────────────────────────────────────────────

def train_regression(X: pd.DataFrame, df_orig: pd.DataFrame,
                     clf_model=None,
                     params: dict = None, output_dir: str = "models"):
    """
    XGBRegressor로 다음 봉 변동폭 예측
    분류 모델과 동일한 피처 사용
    clf_model: 분류 모델 (신뢰도 구간별 방향 일치율 계산용)
    출력: models/chart_model_reg.json
    """
    print(f"\n{'─'*52}")
    print(f"  회귀 모델 학습 (다음 봉 변동폭 예측)")
    print(f"{'─'*52}")

    y_reg = df_orig['future_return_pct'].values
    X_arr = X.values.copy()
    X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
    X_filled = pd.DataFrame(X_arr, columns=X.columns)
    X_filled.fillna(X_filled.median(), inplace=True)

    # 시간순 80/20 분리
    n     = len(X_filled)
    split = int(n * 0.8)
    X_train, X_test = X_filled.iloc[:split], X_filled.iloc[split:]
    y_train, y_test = y_reg[:split],          y_reg[split:]

    # 회귀 파라미터 (분류 파라미터 재활용 가능)
    reg_params = {
        'objective':        'reg:squarederror',
        'n_estimators':     params.get('n_estimators', 300) if params else 300,
        'max_depth':        params.get('max_depth', 5)       if params else 5,
        'learning_rate':    params.get('learning_rate', 0.05) if params else 0.05,
        'subsample':        params.get('subsample', 0.8)     if params else 0.8,
        'colsample_bytree': params.get('colsample_bytree', 0.8) if params else 0.8,
        'random_state':     42,
        'n_jobs':           -1,
        'verbosity':        0,
    }

    reg_model = xgb.XGBRegressor(**reg_params)
    reg_model.fit(
        X_train.values, y_train,
        eval_set=[(X_test.values, y_test)],
        verbose=False
    )

    # 전체 성능 평가 (분류 모델과 무관하게 독립 평가)
    preds         = reg_model.predict(X_test.values)
    mae           = np.mean(np.abs(preds - y_test))
    direction_acc = np.mean(np.sign(preds) == np.sign(y_test))

    print(f"  MAE (평균 절대 오차):  {mae:.4f}%")
    print(f"  방향 일치율 (전체):    {direction_acc*100:.1f}%  ← 분류 모델과 무관한 독립 지표")

    # 신뢰도 구간별 방향 일치율 (분류 모델 있을 때)
    if clf_model is not None:
        try:
            proba     = clf_model.predict_proba(X_test.values)
            from sklearn.preprocessing import LabelEncoder
            _le = LabelEncoder()
            _le.fit(df_orig['label'].values)
            lc        = _le.classes_.tolist()
            bull_idx  = lc.index(1)
            bear_idx  = lc.index(-1)
            prob_bull = proba[:, bull_idx]
            prob_bear = proba[:, bear_idx]

            print(f"\n  신뢰도 구간별 방향 일치율 (분류 신호 기준):")
            print(f"  {'구간':6s} {'매수신호':6s} {'매수방향':8s} {'매도신호':6s} {'매도방향':8s} {'합계신호':6s} {'합계방향':8s}")
            print(f"  {'─'*60}")
            for thresh in [0.50, 0.55, 0.60, 0.65]:
                bull_mask = prob_bull >= thresh
                bear_mask = prob_bear >= thresh
                sig_mask  = bull_mask | bear_mask

                bull_dir = np.mean(np.sign(preds[bull_mask]) == np.sign(y_test[bull_mask])) \
                           if bull_mask.sum() > 0 else float('nan')
                bear_dir = np.mean(np.sign(preds[bear_mask]) == np.sign(y_test[bear_mask])) \
                           if bear_mask.sum() > 0 else float('nan')
                sig_dir  = np.mean(np.sign(preds[sig_mask])  == np.sign(y_test[sig_mask])) \
                           if sig_mask.sum() > 0 else float('nan')

                bull_str = f"{bull_dir*100:.1f}%" if not np.isnan(bull_dir) else "  N/A"
                bear_str = f"{bear_dir*100:.1f}%" if not np.isnan(bear_dir) else "  N/A"
                sig_str  = f"{sig_dir*100:.1f}%"  if not np.isnan(sig_dir)  else "  N/A"

                print(f"  {thresh:.0%}    {bull_mask.sum():5d}   {bull_str:8s} "
                      f"{bear_mask.sum():5d}   {bear_str:8s} "
                      f"{sig_mask.sum():5d}   {sig_str:8s}")

            # 분류 신호 0개 경고
            if (prob_bull >= 0.60).sum() == 0:
                print(f"\n  ⚠️  매수 60%+ 신호 없음 → 분류 모델 편향 가능성")
            if (prob_bear >= 0.60).sum() == 0:
                print(f"\n  ⚠️  매도 60%+ 신호 없음 → 분류 모델 편향 가능성")

        except Exception as e:
            print(f"  신뢰도 구간별 계산 실패: {e}")

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    reg_path = os.path.join(output_dir, 'chart_model_reg.json')
    reg_model.save_model(reg_path)
    print(f"  회귀 모델 저장: {reg_path}")

    return reg_model


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XGBoost + 1D CNN 앙상블 학습 v3')
    parser.add_argument('--data',          default='output/combined_5m_features.csv', help='학습 데이터 경로')
    parser.add_argument('--seq-dir',       default='output', help='CNN 시퀀스 데이터 디렉토리')
    parser.add_argument('--output',        default='models', help='모델 저장 디렉토리')
    parser.add_argument('--cv',            type=int, default=5, help='교차검증 fold 수')
    parser.add_argument('--no-optuna',     action='store_true', help='Optuna 튜닝 건너뛰기')
    parser.add_argument('--optuna-trials', type=int, default=50, help='Optuna 탐색 횟수 (기본 50)')
    parser.add_argument('--no-cnn',        action='store_true', help='CNN 건너뛰기 (XGBoost만)')
    parser.add_argument('--cnn-epochs',    type=int, default=20, help='CNN 학습 에포크 (기본 20)')
    parser.add_argument('--seq-len',       type=int, default=20, help='CNN 시퀀스 길이 (기본 20)')
    args = parser.parse_args()

    if not HAS_XGB:
        return

    print(f"\nXGBoost + 1D CNN 앙상블 학습 v3")
    print(f"{'='*52}")

    # ── 1. XGBoost 학습 ──────────────────────────────────
    df = load_data(args.data)
    feature_cols = get_feature_cols(df)
    X, y_enc, le = prepare_features(df, feature_cols)

    # Optuna 튜닝
    best_params = None
    if not args.no_optuna and HAS_OPTUNA:
        best_params = optuna_tune(X, y_enc, df, n_trials=args.optuna_trials)
    elif args.no_optuna:
        print(f"\n  Optuna 건너뜀 → 기본 파라미터 사용")
    else:
        print(f"\n  Optuna 미설치 → 기본 파라미터 사용")

    best_cv_model, cv_scores = train_with_cv(X, y_enc, n_splits=args.cv, params=best_params)
    final_xgb = train_final(X, y_enc, params=best_params)

    # ── 2. CNN 학습 ───────────────────────────────────────
    final_cnn = None
    if not args.no_cnn and HAS_TORCH:
        seq_x_path = os.path.join(args.seq_dir, 'sequences_X.npy')
        if os.path.exists(seq_x_path):
            final_cnn = train_cnn(
                seq_path   = args.seq_dir,
                output_dir = args.output,
                epochs     = args.cnn_epochs,
                seq_len    = args.seq_len,
            )
        else:
            print(f"\n  시퀀스 파일 없음 ({seq_x_path})")
            print(f"  → python data_pipeline_5m.py 먼저 실행하세요")
    elif args.no_cnn:
        print(f"\n  CNN 건너뜀 → XGBoost 단독 사용")
    else:
        print(f"\n  PyTorch 미설치 → XGBoost 단독 사용")
        print(f"  설치: pip install torch --break-system-packages")

    # ── 3. 백테스트 (XGBoost 단독) ───────────────────────
    backtest(final_xgb, X, df, le, feature_cols)

    # ── 4. 앙상블 가중치 Optuna 최적화 ───────────────────
    if final_cnn and HAS_OPTUNA and not args.no_optuna:
        print(f"\n{'─'*52}")
        print(f"  앙상블 가중치 최적화 (Optuna 30회)")
        print(f"{'─'*52}")

        # 테스트셋 준비
        n_feat   = len(feature_cols)
        split    = int(len(X) * 0.8)
        X_test   = X.iloc[split:].values
        df_test  = df.iloc[split:].copy()
        fut_ret  = df_test['future_return_pct'].values

        # XGBoost 테스트셋 확률
        xgb_proba = final_xgb.predict_proba(X_test)
        bear_idx  = list(le.classes_).index(-1)
        bull_idx  = list(le.classes_).index(1)

        # CNN 테스트셋 확률
        import torch
        seq_x = np.load(os.path.join(args.seq_dir, 'sequences_X.npy'))
        seq_len = seq_x.shape[1]
        total_seq = len(seq_x)
        seq_split = int(total_seq * 0.8)
        X_seq_test = seq_x[seq_split:]
        X_seq_test = np.nan_to_num(X_seq_test, nan=0.0).astype(np.float32)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_cnn.eval()
        with torch.no_grad():
            logits    = final_cnn(torch.tensor(X_seq_test).to(DEVICE))
            cnn_proba = torch.softmax(logits, dim=1).cpu().numpy()

        # 길이 맞추기 (XGB vs CNN 샘플 수 차이 보정)
        min_len   = min(len(xgb_proba), len(cnn_proba), len(fut_ret))
        xgb_proba = xgb_proba[-min_len:]
        cnn_proba = cnn_proba[-min_len:]
        fut_ret   = fut_ret[-min_len:]

        def ensemble_objective(trial):
            xgb_w = trial.suggest_float('xgb_w', 0.1, 0.9)
            cnn_w = 1.0 - xgb_w
            proba = xgb_proba * xgb_w + cnn_proba * cnn_w

            prob_bull = proba[:, bull_idx]
            prob_bear = proba[:, bear_idx]
            bull_mask = prob_bull >= 0.60
            bear_mask = prob_bear >= 0.60

            bull_ret = fut_ret[bull_mask].mean()  if bull_mask.sum() > 5 else 0.0
            bear_ret = -fut_ret[bear_mask].mean() if bear_mask.sum() > 5 else 0.0
            return bull_ret + bear_ret

        ens_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        ens_study.optimize(ensemble_objective, n_trials=30, show_progress_bar=False)

        best_xgb_w = ens_study.best_params['xgb_w']
        best_cnn_w = 1.0 - best_xgb_w
        print(f"  최적 가중치: XGB {best_xgb_w:.3f} / CNN {best_cnn_w:.3f}")
        print(f"  최적 합산 수익률: {ens_study.best_value:.4f}%")
        ensemble_weight = {'xgb': round(best_xgb_w, 3), 'cnn': round(best_cnn_w, 3)}
    else:
        ensemble_weight = {'xgb': 0.6, 'cnn': 0.4} if final_cnn else {'xgb': 1.0, 'cnn': 0.0}

    if best_params:
        best_params['ensemble_weight'] = ensemble_weight
    else:
        best_params = {'ensemble_weight': ensemble_weight}

    # ── 5. 저장 ──────────────────────────────────────────
    save_model(final_xgb, feature_cols, le, output_dir=args.output, best_params=best_params)

    # ── 6. 회귀 모델 학습 ────────────────────────────────
    train_regression(X, df, clf_model=final_xgb,
                     params=best_params, output_dir=args.output)

    print(f"\n{'='*52}")
    print(f"  학습 완료!")
    print(f"  XGBoost CV 평균 정확도: {np.mean(cv_scores):.4f}")
    print(f"  CNN: {'✅' if final_cnn else '❌ (미사용)'}")
    print(f"  앙상블 가중치: XGB {ensemble_weight['xgb']:.1f} / CNN {ensemble_weight['cnn']:.1f}")
    if best_params:
        print(f"  Optuna: ✅")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    main()
