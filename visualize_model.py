"""
모델 학습 결과 시각화 v3
====================
변경사항 (v2 → v3):
  - 경로 chart_ml_v2 → chart_ml_v4

사용법:
  py -3.12 visualize_model.py

출력:
  바탕화면/chart_ml_v4_model_report.png
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
DARK   = '#0a0e17'
CARD   = '#111827'
CARD2  = '#1a2234'
BORDER = '#1f2d45'
GREEN  = '#00d97e'
RED    = '#ff4d6d'
BLUE   = '#4da6ff'
GOLD   = '#ffc043'
GRAY   = '#6b7280'
GRAY2  = '#374151'
WHITE  = '#f0f4ff'
WHITE2 = '#9ca3af'
PURPLE = '#a78bfa'


# ── 경로 설정 ───────────────────────────────────────────────
DESKTOP    = os.path.join(os.path.expanduser('~'), 'Desktop')
BASE_DIR   = os.path.join(DESKTOP, 'chart_ml_v4')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SAVE_PATH  = os.path.join(DESKTOP, 'chart_ml_v4_model_report.png')


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


def get_Xy(df, feature_cols, label_classes):
    feat = [c for c in feature_cols if c in df.columns]
    X = df[feat].copy()
    y = df['label'].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    le = LabelEncoder()
    le.classes_ = np.array(label_classes)
    y_enc = le.transform(y)
    return X, y, y_enc, le, feat


def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=WHITE2, labelsize=8.5, length=3)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.8)


# ── 메인 시각화 ─────────────────────────────────────────────
def main():
    print("모델 결과 시각화 시작...")
    df, model, feature_cols, label_classes = load_everything()
    X, y_raw, y_enc, le, feat = get_Xy(df, feature_cols, label_classes)

    n     = len(X)
    split = int(n * 0.8)
    X_test  = X.iloc[split:].values
    y_test  = y_raw.iloc[split:].values
    df_test = df.iloc[split:].copy()

    proba      = model.predict_proba(X_test)
    preds_enc  = model.predict(X_test)
    preds_orig = le.inverse_transform(preds_enc)

    bear_idx  = list(label_classes).index(-1)
    neut_idx  = list(label_classes).index(0)
    bull_idx  = list(label_classes).index(1)
    prob_bear = proba[:, bear_idx]
    prob_neut = proba[:, neut_idx]
    prob_bull = proba[:, bull_idx]

    acc_overall = (preds_orig == y_test).mean()

    # 누적 수익률 계산
    cum_bull, cum_bear, cum_combined, cum_all = [0], [0], [0], [0]
    for i in range(len(df_test)):
        ret      = df_test['future_return_pct'].iloc[i]
        bull_ret = ret  if prob_bull[i] >= 0.60 else 0
        bear_ret = -ret if prob_bear[i] >= 0.60 else 0
        cum_bull.append(cum_bull[-1] + bull_ret)
        cum_bear.append(cum_bear[-1] + bear_ret)
        cum_combined.append(cum_combined[-1] + bull_ret + bear_ret)
        cum_all.append(cum_all[-1] + ret)

    final_bull = cum_bull[-1]
    final_bear = cum_bear[-1]
    final_comb = cum_combined[-1]
    final_bnh  = cum_all[-1]

    # ── 회귀 모델 로드 및 예측 ────────────────────────────────
    reg_model = None
    reg_path  = os.path.join(MODEL_DIR, 'chart_model_reg.json')
    if os.path.exists(reg_path):
        try:
            import xgboost as xgb_reg
            reg_model = xgb_reg.XGBRegressor()
            reg_model.load_model(reg_path)
            print("  회귀 모델 로드 완료")
        except Exception as e:
            print(f"  회귀 모델 로드 실패: {e}")

    reg_preds    = None
    reg_mae      = None
    reg_dir_all  = None
    reg_dir_60   = None
    if reg_model is not None:
        reg_preds   = reg_model.predict(X_test)
        y_true_reg  = df_test['future_return_pct'].values
        reg_mae     = np.mean(np.abs(reg_preds - y_true_reg))
        reg_dir_all = np.mean(np.sign(reg_preds) == np.sign(y_true_reg))

        # 60%+ 신호 구간 방향 일치율
        sig_60 = (prob_bull >= 0.60) | (prob_bear >= 0.60)
        if sig_60.sum() > 0:
            reg_dir_60 = np.mean(np.sign(reg_preds[sig_60]) == np.sign(y_true_reg[sig_60]))
        else:
            reg_dir_60 = None

        print(f"  회귀 MAE: {reg_mae:.4f}%")
        print(f"  방향 일치율 (전체): {reg_dir_all*100:.1f}%")
        if reg_dir_60 is not None:
            print(f"  방향 일치율 (60%+): {reg_dir_60*100:.1f}%  신호 {sig_60.sum()}개")
        else:
            print(f"  ⚠️  60%+ 신호 없음 → 분류 모델 편향 가능성")

    # ── 레이아웃 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24), facecolor=DARK)
    gs  = gridspec.GridSpec(
        5, 4, figure=fig,
        height_ratios=[0.6, 2.2, 2.2, 2.2, 2.0],
        width_ratios=[1, 1, 1, 1],
        hspace=0.42, wspace=0.32,
        top=0.96, bottom=0.04, left=0.05, right=0.98
    )

    ax_kpi    = fig.add_subplot(gs[0, :])
    ax_cm     = fig.add_subplot(gs[1, 0])
    ax_dist   = fig.add_subplot(gs[1, 1])
    ax_ret    = fig.add_subplot(gs[1, 2])
    ax_conf   = fig.add_subplot(gs[1, 3])
    ax_feat   = fig.add_subplot(gs[2, :3])
    ax_curve2 = fig.add_subplot(gs[2, 3])
    ax_curve  = fig.add_subplot(gs[3, :])
    ax_reg_sc = fig.add_subplot(gs[4, :2])   # 회귀 산점도
    ax_reg_er = fig.add_subplot(gs[4, 2:])   # 회귀 오차 분포

    for ax in [ax_kpi, ax_cm, ax_dist, ax_ret, ax_conf,
               ax_feat, ax_curve2, ax_curve, ax_reg_sc, ax_reg_er]:
        style_ax(ax)

    # ── KPI 헤더 바 ───────────────────────────────────────────
    ax_kpi.set_facecolor(CARD2)
    ax_kpi.axis('off')

    tickers = ', '.join(df['ticker'].unique().tolist()) if 'ticker' in df.columns else ''
    ax_kpi.text(0.01, 0.72, 'XGBoost  5분봉  모델 성능 리포트',
                transform=ax_kpi.transAxes, fontsize=16,
                color=WHITE, fontweight='bold', va='center')
    ax_kpi.text(0.01, 0.22, tickers,
                transform=ax_kpi.transAxes, fontsize=8.5,
                color=GRAY, va='center')

    # KPI 카드 4개
    kpis = [
        ('Hold-out 정확도', f"{acc_overall*100:.1f}%",  WHITE),
        ('합산 수익률',      f"{final_comb:+.2f}%",
         GREEN if final_comb > 0 else RED),
        ('Buy & Hold',     f"{final_bnh:+.2f}%",
         GREEN if final_bnh > 0 else RED),
        ('테스트 샘플',     f"{len(y_test):,}개",        BLUE),
    ]
    for i, (label, val, clr) in enumerate(kpis):
        bx = 0.38 + i * 0.155
        ax_kpi.text(bx, 0.72, val,
                    transform=ax_kpi.transAxes, fontsize=15,
                    color=clr, fontweight='bold', va='center', ha='center')
        ax_kpi.text(bx, 0.22, label,
                    transform=ax_kpi.transAxes, fontsize=8,
                    color=GRAY, va='center', ha='center')
        if i < 3:
            ax_kpi.plot([bx + 0.072, bx + 0.072], [0.1, 0.9],
                        color=BORDER, linewidth=0.8,
                        transform=ax_kpi.transAxes)

    # ── ① 혼동행렬 ────────────────────────────────────────────
    labels_order = [-1, 0, 1]
    cm      = confusion_matrix(y_test, preds_orig, labels=labels_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap_custom = plt.cm.get_cmap('Blues')
    ax_cm.imshow(cm_norm, cmap=cmap_custom, vmin=0, vmax=1, aspect='auto')
    ax_cm.set_xticks([0, 1, 2])
    ax_cm.set_yticks([0, 1, 2])
    ax_cm.set_xticklabels(LABELS_KO, color=WHITE2, fontsize=10)
    ax_cm.set_yticklabels(LABELS_KO, color=WHITE2, fontsize=10)
    ax_cm.set_title('혼동행렬', color=WHITE, fontsize=11, pad=10, fontweight='bold')
    ax_cm.set_xlabel('예측', color=GRAY, fontsize=9)
    ax_cm.set_ylabel('실제', color=GRAY, fontsize=9)
    for i in range(3):
        for j in range(3):
            val = cm_norm[i, j]
            clr = WHITE if val > 0.55 else WHITE2
            ax_cm.text(j, i, f'{val:.2f}\n({cm[i,j]:,})',
                       ha='center', va='center', color=clr, fontsize=8.5,
                       fontweight='bold' if i == j else 'normal')

    # 대각선 테두리 강조
    for k in range(3):
        ax_cm.add_patch(plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor=GOLD, linewidth=1.5
        ))

    # ── ② 신호 분포 ───────────────────────────────────────────
    label_counts = pd.Series(y_raw).value_counts().reindex([-1, 0, 1], fill_value=0)
    colors_dist  = [RED, GOLD, GREEN]
    bars = ax_dist.bar(
        LABELS_KO, label_counts.values,
        color=colors_dist, alpha=0.85,
        edgecolor=DARK, linewidth=0.5,
        width=0.55
    )
    total = label_counts.sum()
    for bar, cnt, clr in zip(bars, label_counts.values, colors_dist):
        ax_dist.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.008,
            f'{cnt:,}\n{cnt/total*100:.1f}%',
            ha='center', va='bottom', color=clr, fontsize=8.5, fontweight='bold'
        )
    ax_dist.set_title('신호 분포', color=WHITE, fontsize=11, pad=10, fontweight='bold')
    ax_dist.set_ylabel('샘플 수', color=GRAY, fontsize=9)
    ax_dist.set_ylim(0, label_counts.max() * 1.25)
    ax_dist.tick_params(colors=WHITE2)
    ax_dist.yaxis.set_tick_params(labelcolor=WHITE2)

    # ── ③ 신뢰도별 평균 수익률 ────────────────────────────────
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    bull_rets, bear_rets = [], []
    for thresh in thresholds:
        bm = prob_bull >= thresh
        dm = prob_bear >= thresh
        bull_rets.append(df_test['future_return_pct'].values[bm].mean() if bm.sum() > 0 else 0)
        bear_rets.append(-df_test['future_return_pct'].values[dm].mean() if dm.sum() > 0 else 0)

    x_thr = np.arange(len(thresholds))
    w     = 0.32
    b1 = ax_ret.bar(x_thr - w/2, bull_rets, w, color=GREEN, alpha=0.8,
                    label='매수', edgecolor=DARK, linewidth=0.5)
    b2 = ax_ret.bar(x_thr + w/2, bear_rets, w, color=RED,   alpha=0.8,
                    label='매도', edgecolor=DARK, linewidth=0.5)
    ax_ret.axhline(0, color=GRAY2, linewidth=0.8)
    ax_ret.set_xticks(x_thr)
    ax_ret.set_xticklabels([f'{int(t*100)}%' for t in thresholds], color=WHITE2, fontsize=8)
    ax_ret.set_title('신뢰도별 평균 수익률', color=WHITE, fontsize=11, pad=10, fontweight='bold')
    ax_ret.set_xlabel('신뢰도 임계값', color=GRAY, fontsize=9)
    ax_ret.set_ylabel('평균 수익률 (%)', color=GRAY, fontsize=9)
    ax_ret.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=WHITE2, fontsize=8)
    ax_ret.yaxis.set_tick_params(labelcolor=WHITE2)

    # 최고 수익률 임계값 강조
    best_bull_idx = int(np.argmax(bull_rets))
    best_bear_idx = int(np.argmax(bear_rets))
    ax_ret.bar(best_bull_idx - w/2, bull_rets[best_bull_idx], w,
               color=GREEN, alpha=1.0, edgecolor=WHITE, linewidth=1.2)
    ax_ret.bar(best_bear_idx + w/2, bear_rets[best_bear_idx], w,
               color=RED,   alpha=1.0, edgecolor=WHITE, linewidth=1.2)

    # ── ④ 신뢰도 분포 ─────────────────────────────────────────
    max_proba = proba.max(axis=1)
    n_bins    = 35
    counts, edges = np.histogram(max_proba, bins=n_bins, range=(0.3, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [GREEN if c > 0.6 else (GOLD if c > 0.5 else BLUE) for c in centers]
    ax_conf.bar(centers, counts, width=(edges[1]-edges[0])*0.85,
                color=bar_colors, alpha=0.85, edgecolor=DARK, linewidth=0.4)
    ax_conf.axvline(max_proba.mean(), color=GOLD, linewidth=1.5, linestyle='--',
                    label=f'평균 {max_proba.mean():.2f}', alpha=0.9)
    ax_conf.axvline(0.6, color=WHITE, linewidth=1.0, linestyle=':',
                    label='기준 60%', alpha=0.6)
    ax_conf.set_title('신뢰도 분포', color=WHITE, fontsize=11, pad=10, fontweight='bold')
    ax_conf.set_xlabel('최대 확률값', color=GRAY, fontsize=9)
    ax_conf.set_ylabel('빈도', color=GRAY, fontsize=9)
    ax_conf.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=WHITE2, fontsize=8)
    ax_conf.yaxis.set_tick_params(labelcolor=WHITE2)
    ax_conf.xaxis.set_tick_params(labelcolor=WHITE2)

    # ── ⑤ 피처 중요도 Top 15 ──────────────────────────────────
    importances = model.feature_importances_
    feat_imp    = sorted(zip(feat, importances), key=lambda x: x[1])[-15:]
    feat_names, feat_vals = zip(*feat_imp)
    max_v = max(feat_vals)

    # 그라데이션 색상 (중요도 높을수록 밝게)
    bar_colors_feat = []
    for v in feat_vals:
        ratio = v / max_v
        if ratio >= 0.7:   bar_colors_feat.append(GOLD)
        elif ratio >= 0.4: bar_colors_feat.append(BLUE)
        else:              bar_colors_feat.append(GRAY2)

    y_pos = np.arange(len(feat_names))
    bars_feat = ax_feat.barh(
        y_pos, feat_vals,
        color=bar_colors_feat, alpha=0.88,
        edgecolor=DARK, linewidth=0.4, height=0.7
    )
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(feat_names, color=WHITE2, fontsize=8.5)
    ax_feat.set_title('피처 중요도  Top 15', color=WHITE, fontsize=11, pad=10, fontweight='bold')
    ax_feat.set_xlabel('Importance', color=GRAY, fontsize=9)
    ax_feat.xaxis.set_tick_params(labelcolor=WHITE2)

    # 값 라벨
    for i, val in enumerate(feat_vals):
        ax_feat.text(val + max_v * 0.005, i, f'{val:.4f}',
                     va='center', color=WHITE2, fontsize=7.5)

    # 1위 강조 박스
    ax_feat.barh(y_pos[-1], feat_vals[-1], color=GOLD, alpha=1.0,
                 edgecolor=WHITE, linewidth=1.0, height=0.7)

    # ── ⑥ 신호별 최종 수익률 요약 (소형) ─────────────────────
    ax_curve2.axis('off')
    ax_curve2.set_facecolor(CARD2)

    summary = [
        ('▲ 매수 단독',  f'{final_bull:+.2f}%', GREEN),
        ('▼ 매도 단독',  f'{final_bear:+.2f}%', RED),
        ('합산',         f'{final_comb:+.2f}%', BLUE),
        ('Buy & Hold',  f'{final_bnh:+.2f}%',  GRAY),
    ]
    ax_curve2.text(0.5, 0.93, '최종 누적 수익률',
                   transform=ax_curve2.transAxes, fontsize=10,
                   color=WHITE, ha='center', va='top', fontweight='bold')

    for i, (lbl, val, clr) in enumerate(summary):
        y = 0.76 - i * 0.18
        ax_curve2.text(0.08, y, lbl,
                       transform=ax_curve2.transAxes, fontsize=9,
                       color=GRAY, va='center')
        ax_curve2.text(0.92, y, val,
                       transform=ax_curve2.transAxes, fontsize=12,
                       color=clr, va='center', ha='right', fontweight='bold')
        ax_curve2.plot([0.05, 0.95], [y - 0.07, y - 0.07], color=BORDER, linewidth=0.6,
                       transform=ax_curve2.transAxes)

    # ── ⑦ 누적 수익률 곡선 ────────────────────────────────────
    x_c = np.arange(len(cum_bull))

    ax_curve.fill_between(x_c, cum_combined, 0,
                           where=[v > 0 for v in cum_combined],
                           alpha=0.07, color=BLUE, zorder=1)
    ax_curve.fill_between(x_c, cum_combined, 0,
                           where=[v < 0 for v in cum_combined],
                           alpha=0.07, color=RED, zorder=1)

    ax_curve.plot(x_c, cum_bull,     color=GREEN, linewidth=1.3,
                  label=f'▲ 매수 신호 (60%+)  {final_bull:+.2f}%', alpha=0.9, zorder=3)
    ax_curve.plot(x_c, cum_bear,     color=RED,   linewidth=1.3,
                  label=f'▼ 매도 신호 (60%+)  {final_bear:+.2f}%', alpha=0.9, zorder=3)
    ax_curve.plot(x_c, cum_combined, color=BLUE,  linewidth=2.2,
                  label=f'합산  {final_comb:+.2f}%', alpha=1.0, zorder=4)
    ax_curve.plot(x_c, cum_all,      color=GRAY,  linewidth=0.9,
                  label=f'Buy & Hold  {final_bnh:+.2f}%', alpha=0.5,
                  linestyle='--', zorder=2)

    ax_curve.axhline(0, color=GRAY2, linewidth=0.8, zorder=1)
    ax_curve.axhline(final_comb, color=BLUE, linewidth=0.6,
                     linestyle=':', alpha=0.4, zorder=2)
    ax_curve.text(len(x_c) - 1, final_comb,
                  f'  {final_comb:+.2f}%',
                  color=BLUE, fontsize=8.5, va='center', alpha=0.8)

    ax_curve.set_title(
        '누적 수익률 곡선  (Hold-out 테스트셋  |  신뢰도 60% 이상)',
        color=WHITE, fontsize=11, pad=10, fontweight='bold'
    )
    ax_curve.set_xlabel('봉 인덱스', color=GRAY, fontsize=9)
    ax_curve.set_ylabel('누적 수익률 (%)', color=GRAY, fontsize=9)
    ax_curve.legend(
        facecolor=CARD2, edgecolor=BORDER, labelcolor=WHITE2,
        fontsize=9, loc='upper left', framealpha=0.9
    )
    ax_curve.yaxis.set_tick_params(labelcolor=WHITE2)
    ax_curve.xaxis.set_tick_params(labelcolor=WHITE2)

    # ── ⑧ 회귀 모델 성능 차트 ────────────────────────────────
    if reg_preds is not None:
        y_true = df_test['future_return_pct'].values

        # 신뢰도 60%+ 구간 마스크
        bull_60 = prob_bull >= 0.60
        bear_60 = prob_bear >= 0.60
        sig_60  = bull_60 | bear_60

        # 60%+ 방향 일치율 (None 처리)
        dir_60_bull = np.mean(np.sign(reg_preds[bull_60]) == np.sign(y_true[bull_60])) \
                      if bull_60.sum() > 0 else None
        dir_60_bear = np.mean(np.sign(reg_preds[bear_60]) == np.sign(y_true[bear_60])) \
                      if bear_60.sum() > 0 else None
        dir_60_all  = np.mean(np.sign(reg_preds[sig_60])  == np.sign(y_true[sig_60])) \
                      if sig_60.sum() > 0 else None

        # 타이틀 구성
        title_dir_all = f"전체={reg_dir_all*100:.1f}%"
        title_dir_60  = f"60%+={dir_60_all*100:.1f}%" if dir_60_all is not None else "60%+=신호없음"

        # 산점도
        ax_reg_sc.scatter(y_true, reg_preds,
                          alpha=0.08, s=3, color=GRAY, zorder=1)
        if bull_60.sum() > 0:
            lbl = f'매수 60%+ ({bull_60.sum()}개, 방향일치 {dir_60_bull*100:.1f}%)'
            ax_reg_sc.scatter(y_true[bull_60], reg_preds[bull_60],
                              alpha=0.5, s=8, color=GREEN, zorder=3, label=lbl)
        if bear_60.sum() > 0:
            lbl = f'매도 60%+ ({bear_60.sum()}개, 방향일치 {dir_60_bear*100:.1f}%)'
            ax_reg_sc.scatter(y_true[bear_60], reg_preds[bear_60],
                              alpha=0.5, s=8, color=RED, zorder=3, label=lbl)
        if sig_60.sum() == 0:
            ax_reg_sc.text(0.5, 0.5, '⚠️ 60%+ 신호 없음\n분류 모델 편향 확인 필요',
                           transform=ax_reg_sc.transAxes, color=GOLD,
                           ha='center', va='center', fontsize=10)

        lim = max(abs(y_true).max(), abs(reg_preds).max()) * 1.1
        ax_reg_sc.plot([-lim, lim], [-lim, lim],
                       color=GOLD, linewidth=1.0, linestyle='--',
                       alpha=0.7, zorder=2, label='이상적 예측')
        ax_reg_sc.axhline(0, color=GRAY2, linewidth=0.6)
        ax_reg_sc.axvline(0, color=GRAY2, linewidth=0.6)
        ax_reg_sc.set_xlim(-lim, lim)
        ax_reg_sc.set_ylim(-lim, lim)
        ax_reg_sc.set_title(
            f'회귀 모델  실제 vs 예측  (방향일치: {title_dir_all}  {title_dir_60})',
            color=WHITE, fontsize=11, pad=10, fontweight='bold'
        )
        ax_reg_sc.set_xlabel('실제 변동폭 (%)', color=GRAY, fontsize=9)
        ax_reg_sc.set_ylabel('예측 변동폭 (%)', color=GRAY, fontsize=9)
        ax_reg_sc.legend(facecolor=CARD2, edgecolor=BORDER,
                         labelcolor=WHITE2, fontsize=8)
        ax_reg_sc.yaxis.set_tick_params(labelcolor=WHITE2)
        ax_reg_sc.xaxis.set_tick_params(labelcolor=WHITE2)

        # 오차 분포
        errors_all = reg_preds - y_true
        ax_reg_er.hist(errors_all, bins=50, color=GRAY, alpha=0.5,
                       edgecolor=DARK, linewidth=0.3, label='전체')
        if sig_60.sum() > 0:
            errors_60 = reg_preds[sig_60] - y_true[sig_60]
            ax_reg_er.hist(errors_60, bins=50, color=BLUE, alpha=0.75,
                           edgecolor=DARK, linewidth=0.3, label='신뢰도 60%+')
            ax_reg_er.axvline(errors_60.mean(), color=RED, linewidth=1.2,
                              linestyle=':',
                              label=f'60%+ 평균오차 {errors_60.mean():.4f}%', alpha=0.8)
        ax_reg_er.axvline(0, color=GOLD, linewidth=1.5,
                          linestyle='--', label='오차 0', alpha=0.9)
        ax_reg_er.set_title('회귀 예측 오차 분포  (전체 vs 신뢰도 60%+)',
                             color=WHITE, fontsize=11, pad=10, fontweight='bold')
        ax_reg_er.set_xlabel('예측 오차 (%)', color=GRAY, fontsize=9)
        ax_reg_er.set_ylabel('빈도', color=GRAY, fontsize=9)
        ax_reg_er.legend(facecolor=CARD2, edgecolor=BORDER,
                         labelcolor=WHITE2, fontsize=8)
        ax_reg_er.yaxis.set_tick_params(labelcolor=WHITE2)
        ax_reg_er.xaxis.set_tick_params(labelcolor=WHITE2)
    else:
        for ax in [ax_reg_sc, ax_reg_er]:
            ax.axis('off')
            ax.text(0.5, 0.5, '회귀 모델 없음\ntrain_model.py 재실행 필요',
                    transform=ax.transAxes, color=GRAY,
                    ha='center', va='center', fontsize=11)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight',
                facecolor=DARK, edgecolor='none')
    plt.close()
    print(f"\n저장 완료: {SAVE_PATH}")


if __name__ == '__main__':
    main()
