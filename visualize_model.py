"""
모델 학습 결과 시각화
====================
사용법:
  py -3.12 visualize_model.py

출력:
  바탕화면/chart_ml_model_report.png
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ── 한글 폰트 설정 (Windows Malgun Gothic 강제 지정) ────────
def set_korean_font():
    font_paths = [
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/malgunbd.ttf',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            fe = fm.FontEntry(fname=fp, name='Malgun Gothic')
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  폰트 적용: Malgun Gothic")
            return
    for font in ['NanumGothic', 'AppleGothic', 'DejaVu Sans']:
        if any(f.name == font for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return

set_korean_font()

# ── 레이블 한글 매핑 ────────────────────────────────────────
LABELS_KO   = ['매도', '관망', '매수']   # -1, 0, 1 순서
LABEL_MAP   = {-1: '매도', 0: '관망', 1: '매수'}

# ── 색상 팔레트 ─────────────────────────────────────────────
DARK  = '#0d1117'
CARD  = '#161b22'
GREEN = '#2ea043'
RED   = '#f85149'
BLUE  = '#58a6ff'
GOLD  = '#e3b341'
GRAY  = '#8b949e'
WHITE = '#e6edf3'

# ── 경로 설정 ───────────────────────────────────────────────
DESKTOP    = os.path.join(os.path.expanduser('~'), 'Desktop')
BASE_DIR   = os.path.join(DESKTOP, 'chart_ml')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SAVE_PATH  = os.path.join(DESKTOP, 'chart_ml_model_report.png')


# ── 데이터 & 모델 로드 ──────────────────────────────────────
def load_everything():
    data_path = os.path.join(OUTPUT_DIR, 'combined_5m_features.csv')
    if not os.path.exists(data_path):
        csvs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_5m_features.csv')]
        if not csvs:
            raise FileNotFoundError(f"CSV 없음: {OUTPUT_DIR}")
        data_path = os.path.join(OUTPUT_DIR, csvs[0])

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODEL_DIR, 'chart_model.json'))
    except Exception as e:
        raise RuntimeError(f"모델 로드 실패: {e}")

    with open(os.path.join(MODEL_DIR, 'feature_list.txt')) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    with open(os.path.join(MODEL_DIR, 'label_classes.json')) as f:
        label_classes = json.load(f)

    return df, model, feature_cols, label_classes


def get_Xy(df, feature_cols):
    feat = [c for c in feature_cols if c in df.columns]
    X = df[feat].copy()
    y = df['label'].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y, y_enc, le, feat


# ── 메인 시각화 ─────────────────────────────────────────────
def main():
    print("모델 결과 시각화 시작...")
    df, model, feature_cols, label_classes = load_everything()
    X, y_raw, y_enc, le, feat = get_Xy(df, feature_cols)

    n     = len(X)
    split = int(n * 0.8)
    X_test  = X.iloc[split:].values
    y_test  = y_raw.iloc[split:].values
    df_test = df.iloc[split:].copy()

    proba      = model.predict_proba(X_test)
    preds_enc  = model.predict(X_test)
    preds_orig = le.inverse_transform(preds_enc)

    prob_bear = proba[:, 0]
    prob_neut = proba[:, 1]
    prob_bull = proba[:, 2]

    # ── 레이아웃 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor=DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                            top=0.94, bottom=0.04, left=0.06, right=0.97)

    ax_title = fig.add_subplot(gs[0, :])
    ax_cm    = fig.add_subplot(gs[1, 0])
    ax_dist  = fig.add_subplot(gs[1, 1])
    ax_ret   = fig.add_subplot(gs[1, 2])
    ax_feat  = fig.add_subplot(gs[2, :2])
    ax_conf  = fig.add_subplot(gs[2, 2])
    ax_curve = fig.add_subplot(gs[3, :])

    for ax in [ax_title, ax_cm, ax_dist, ax_ret, ax_feat, ax_conf, ax_curve]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # ① 타이틀
    ax_title.axis('off')
    ticker_info = f"  |  종목: {', '.join(df['ticker'].unique().tolist())}" if 'ticker' in df.columns else ''
    ax_title.text(0.5, 0.65, 'XGBoost 5분봉 모델 성능 리포트',
                  ha='center', va='center', fontsize=20, color=WHITE,
                  fontweight='bold', transform=ax_title.transAxes)
    acc_overall = (preds_orig == y_test).mean()
    ax_title.text(0.5, 0.2,
                  f"Hold-out 정확도: {acc_overall*100:.1f}%   |   테스트 샘플: {len(y_test):,}개{ticker_info}",
                  ha='center', va='center', fontsize=12, color=GRAY,
                  transform=ax_title.transAxes)

    # ② 혼동행렬
    labels_order = [-1, 0, 1]
    cm      = confusion_matrix(y_test, preds_orig, labels=labels_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax_cm.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax_cm.set_xticks([0, 1, 2])
    ax_cm.set_yticks([0, 1, 2])
    ax_cm.set_xticklabels(LABELS_KO, color=WHITE, fontsize=10)
    ax_cm.set_yticklabels(LABELS_KO, color=WHITE, fontsize=10)
    ax_cm.set_title('혼동행렬 (Confusion Matrix)', color=WHITE, fontsize=11, pad=10)
    ax_cm.set_xlabel('예측값', color=GRAY, fontsize=9)
    ax_cm.set_ylabel('실제값', color=GRAY, fontsize=9)
    for i in range(3):
        for j in range(3):
            val = cm_norm[i, j]
            ax_cm.text(j, i, f'{val:.2f}\n({cm[i,j]})',
                       ha='center', va='center',
                       color='white' if val > 0.5 else WHITE, fontsize=8)

    # ③ 라벨 분포
    label_counts = pd.Series(y_raw).value_counts().reindex([-1, 0, 1], fill_value=0)
    bars = ax_dist.bar(LABELS_KO, label_counts.values,
                       color=[RED, GOLD, GREEN], alpha=0.85, edgecolor='#30363d')
    ax_dist.set_title('신호 분포', color=WHITE, fontsize=11, pad=10)
    ax_dist.set_ylabel('샘플 수', color=GRAY, fontsize=9)
    total = label_counts.sum()
    for bar, cnt in zip(bars, label_counts.values):
        ax_dist.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + total * 0.01,
                     f'{cnt:,}\n({cnt/total*100:.1f}%)',
                     ha='center', va='bottom', color=WHITE, fontsize=8)
    ax_dist.set_ylim(0, label_counts.max() * 1.2)

    # ④ 신뢰도 임계값별 수익률
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    bull_rets, bear_rets = [], []
    for thresh in thresholds:
        bm = prob_bull >= thresh
        dm = prob_bear >= thresh
        bull_rets.append(df_test['future_return_pct'].values[bm].mean() if bm.sum() > 0 else 0)
        bear_rets.append(-df_test['future_return_pct'].values[dm].mean() if dm.sum() > 0 else 0)

    x_thr = np.arange(len(thresholds))
    w = 0.35
    ax_ret.bar(x_thr - w/2, bull_rets, w, color=GREEN, alpha=0.8, label='매수 신호')
    ax_ret.bar(x_thr + w/2, bear_rets, w, color=RED,   alpha=0.8, label='매도 신호')
    ax_ret.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax_ret.set_xticks(x_thr)
    ax_ret.set_xticklabels([f'{int(t*100)}%' for t in thresholds], color=WHITE, fontsize=8)
    ax_ret.set_title('신뢰도 임계값별 평균 수익률', color=WHITE, fontsize=11, pad=10)
    ax_ret.set_xlabel('신뢰도 임계값', color=GRAY, fontsize=9)
    ax_ret.set_ylabel('평균 수익률 (%)', color=GRAY, fontsize=9)
    ax_ret.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=WHITE, fontsize=8)

    # ⑤ 피처 중요도
    importances = model.feature_importances_
    feat_imp    = sorted(zip(feat, importances), key=lambda x: x[1])[-15:]
    feat_names, feat_vals = zip(*feat_imp)
    colors_feat = [BLUE if v < max(feat_vals)*0.6 else GOLD for v in feat_vals]
    y_pos = np.arange(len(feat_names))
    ax_feat.barh(y_pos, feat_vals, color=colors_feat, alpha=0.85, edgecolor='#30363d')
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(feat_names, color=WHITE, fontsize=8)
    ax_feat.set_title('피처 중요도 Top 15', color=WHITE, fontsize=11, pad=10)
    ax_feat.set_xlabel('Importance', color=GRAY, fontsize=9)
    for i, val in enumerate(feat_vals):
        ax_feat.text(val + max(feat_vals)*0.01, i, f'{val:.4f}',
                     va='center', color=GRAY, fontsize=7)

    # ⑥ 신뢰도 분포
    max_proba = proba.max(axis=1)
    ax_conf.hist(max_proba, bins=30, color=BLUE, alpha=0.75, edgecolor='#30363d')
    ax_conf.axvline(max_proba.mean(), color=GOLD, linewidth=1.5, linestyle='--',
                    label=f'평균 {max_proba.mean():.2f}')
    ax_conf.axvline(0.6, color=RED, linewidth=1.2, linestyle=':',
                    label='임계값 60%')
    ax_conf.set_title('예측 신뢰도 분포', color=WHITE, fontsize=11, pad=10)
    ax_conf.set_xlabel('최대 확률값', color=GRAY, fontsize=9)
    ax_conf.set_ylabel('빈도', color=GRAY, fontsize=9)
    ax_conf.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=WHITE, fontsize=8)

    # ⑦ 누적 수익률 곡선
    cum_bull, cum_bear, cum_all = [0], [0], [0]
    for i in range(len(df_test)):
        ret = df_test['future_return_pct'].iloc[i]
        cum_bull.append(cum_bull[-1] + (ret  if prob_bull[i] >= 0.60 else 0))
        cum_bear.append(cum_bear[-1] + (-ret if prob_bear[i] >= 0.60 else 0))
        cum_all.append(cum_all[-1] + ret)

    x_c = np.arange(len(cum_bull))
    ax_curve.plot(x_c, cum_bull, color=GREEN, linewidth=1.2, label='매수 신호 (신뢰도 60%+)', alpha=0.9)
    ax_curve.plot(x_c, cum_bear, color=RED,   linewidth=1.2, label='매도 신호 숏 (신뢰도 60%+)', alpha=0.9)
    ax_curve.plot(x_c, cum_all,  color=GRAY,  linewidth=0.8, label='Buy & Hold', alpha=0.6, linestyle='--')
    ax_curve.axhline(0, color='#30363d', linewidth=0.8)
    ax_curve.fill_between(x_c, cum_bull, 0, where=[v > 0 for v in cum_bull], alpha=0.08, color=GREEN)
    ax_curve.fill_between(x_c, cum_bull, 0, where=[v < 0 for v in cum_bull], alpha=0.08, color=RED)
    ax_curve.set_title('누적 수익률 곡선 (Hold-out 테스트셋, 신뢰도 60% 이상)', color=WHITE, fontsize=11, pad=10)
    ax_curve.set_xlabel('봉 인덱스', color=GRAY, fontsize=9)
    ax_curve.set_ylabel('누적 수익률 (%)', color=GRAY, fontsize=9)
    ax_curve.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=WHITE, fontsize=9)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"\n저장 완료: {SAVE_PATH}")


if __name__ == '__main__':
    main()
