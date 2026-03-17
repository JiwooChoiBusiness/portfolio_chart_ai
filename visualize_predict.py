"""
예측 결과 시각화 v3
================
변경사항 (v2 → v3):
  - 경로 chart_ml_v2 → chart_ml_v4

사용법:
  py -3.12 visualize_predict.py --ticker AAPL
  py -3.12 visualize_predict.py --ticker NVDA --bars 30

출력:
  바탕화면/AAPL_predict_chart.png
"""

import argparse
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
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

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
            return
    for font in ['NanumGothic', 'AppleGothic', 'DejaVu Sans']:
        if any(f.name == font for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return

set_korean_font()

# ── 레이블 & 강도 한글 매핑 ─────────────────────────────────
PRED_KO = {
    'BULLISH': '매수',
    'BEARISH': '매도',
    'NEUTRAL': '관망',
}
STRENGTH_KO = {
    '강한 상승': '강한 상승세',
    '약한 상승': '약한 상승세',
    '강한 하락': '강한 하락세',
    '약한 하락': '약한 하락세',
    '불확실':   '방향 미정',
    '횡보':     '방향 미정',
}

# ── 색상 ────────────────────────────────────────────────────
DARK       = '#0a0e17'
CARD       = '#111827'
CARD2      = '#1a2234'
BORDER     = '#1f2d45'
GREEN      = '#00d97e'
GREEN_DIM  = '#00d97e55'
RED        = '#ff4d6d'
RED_DIM    = '#ff4d6d55'
BLUE       = '#4da6ff'
BLUE_DIM   = '#4da6ff33'
GOLD       = '#ffc043'
GOLD_DIM   = '#ffc04333'
GRAY       = '#6b7280'
GRAY2      = '#374151'
WHITE      = '#f0f4ff'
WHITE2     = '#9ca3af'
BULL_BODY  = '#00d97e'
BEAR_BODY  = '#ff4d6d'
BULL_WICK  = '#00b868'
BEAR_WICK  = '#cc3d57'

# ── 경로 ────────────────────────────────────────────────────
DESKTOP   = os.path.join(os.path.expanduser('~'), 'Desktop')
BASE_DIR  = os.path.join(DESKTOP, 'chart_ml_v4')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# ── 데이터 수집 & 예측 ──────────────────────────────────────
def fetch_and_predict(ticker: str, n_bars: int = 30):
    import sys
    sys.path.insert(0, BASE_DIR)
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
    import xgboost as xgb
    import yfinance as yf

    print(f"  [{ticker}] 데이터 수집 중...")
    df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[['open','high','low','close','volume']].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York')
    df = df.iloc[df.index.indexer_between_time('09:30','15:55')]
    df.dropna(inplace=True)

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

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, 'chart_model.json'))
    with open(os.path.join(MODEL_DIR, 'feature_list.txt')) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    with open(os.path.join(MODEL_DIR, 'label_classes.json')) as f:
        label_classes = json.load(f)

    # 회귀 모델 로드 (있을 때만)
    reg_model = None
    reg_path  = os.path.join(MODEL_DIR, 'chart_model_reg.json')
    if os.path.exists(reg_path):
        reg_model = xgb.XGBRegressor()
        reg_model.load_model(reg_path)

    n_predict = min(n_bars, len(df))
    X = pd.DataFrame(index=df.index[-n_predict:])
    for col in feature_cols:
        X[col] = df[col].iloc[-n_predict:].values if col in df.columns else 0.0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    proba     = model.predict_proba(X.values)
    preds_enc = model.predict(X.values)

    # 회귀 예측
    reg_preds = reg_model.predict(X.values) if reg_model is not None else None

    results = []
    for i in range(n_predict):
        prob_map  = {int(label_classes[j]): float(proba[i][j]) for j in range(len(label_classes))}
        prob_bull = prob_map.get(1,  0.0)
        prob_neut = prob_map.get(0,  0.0)
        prob_bear = prob_map.get(-1, 0.0)
        pred_orig = int(label_classes[int(preds_enc[i])])
        max_prob  = max(prob_bull, prob_bear, prob_neut)

        if pred_orig == 1:
            direction = 'BULLISH'
            strength  = '강한 상승' if prob_bull >= 0.65 else ('약한 상승' if prob_bull >= 0.50 else '불확실')
        elif pred_orig == -1:
            direction = 'BEARISH'
            strength  = '강한 하락' if prob_bear >= 0.65 else ('약한 하락' if prob_bear >= 0.50 else '불확실')
        else:
            direction = 'NEUTRAL'
            strength  = '횡보'

        row = df.iloc[-(n_predict - i)]

        # 실제 다음봉 결과 계산 (마지막 봉은 아직 모름)
        actual_next = None
        actual_ret  = None
        df_idx = len(df) - n_predict + i
        if df_idx + 1 < len(df):
            next_close = float(df['close'].iloc[df_idx + 1])
            curr_close = float(row['close'])
            ret = (next_close / curr_close - 1) * 100
            actual_ret = ret
            if ret > 0.15:    actual_next = 'BULLISH'
            elif ret < -0.15: actual_next = 'BEARISH'
            else:              actual_next = 'NEUTRAL'

        # 회귀 예측 (변동폭 → 종가)
        curr_close  = float(row['close'])
        pred_change = float(reg_preds[i]) if reg_preds is not None else None
        pred_close  = round(curr_close * (1 + pred_change / 100), 2) if pred_change is not None else None

        results.append({
            'timestamp':   X.index[i],
            'open':        float(row['open']),
            'high':        float(row['high']),
            'low':         float(row['low']),
            'close':       curr_close,
            'volume':      float(row['volume']),
            'rsi14':       float(row['rsi14'])     if 'rsi14'     in df.columns else 50.0,
            'macd_hist':   float(row['macd_hist']) if 'macd_hist' in df.columns else 0.0,
            'vwap':        float(row['vwap'])      if 'vwap'      in df.columns else curr_close,
            'dist_vwap':   float(row['dist_vwap']) if 'dist_vwap' in df.columns else 0.0,
            'prediction':  direction,
            'pred_ko':     PRED_KO[direction],
            'strength':    STRENGTH_KO.get(strength, strength),
            'prob_bull':   round(prob_bull * 100, 1),
            'prob_bear':   round(prob_bear * 100, 1),
            'prob_neut':   round(prob_neut * 100, 1),
            'confidence':  round(max_prob * 100, 1),
            'actual_next': actual_next,
            'actual_ret':  actual_ret,
            'pred_change': round(pred_change, 3) if pred_change is not None else None,
            'pred_close':  pred_close,
        })

    return results


# ── 캔들 그리기 ─────────────────────────────────────────────
def draw_candles(ax, results, bb_upper=None, bb_lower=None):
    """
    신뢰도에 따라 신호 마커 크기/색상 강도 변화
    볼린저 밴드 오버레이
    """
    for i, r in enumerate(results):
        o, h, l, c = r['open'], r['high'], r['low'], r['close']
        is_bull = c >= o
        body_color = BULL_BODY if is_bull else BEAR_BODY
        wick_color = BULL_WICK if is_bull else BEAR_WICK
        body_b = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.015)

        # 심지
        ax.plot([i, i], [l, h], color=wick_color, linewidth=0.8, alpha=0.85, zorder=2)
        # 몸통
        ax.bar(i, body_h, bottom=body_b, width=0.65,
               color=body_color, alpha=0.9, edgecolor='none', zorder=3)

        # 신호 마커 — 신뢰도에 따라 크기/투명도 변화
        pred = r['prediction']
        conf = r['confidence']
        if pred in ('BULLISH', 'BEARISH') and conf >= 55:
            alpha     = min(0.4 + (conf - 55) / 45 * 0.6, 1.0)
            span_clr  = GREEN if pred == 'BULLISH' else RED
            ax.axvspan(i - 0.5, i + 0.5, alpha=alpha * 0.12, color=span_clr, zorder=1)

            # 삼각형 마커
            marker_size = 6 + (conf - 55) / 10
            if pred == 'BULLISH':
                ax.plot(i, l * 0.9994, marker='^', markersize=marker_size,
                        color=GREEN, alpha=alpha, zorder=5, markeredgewidth=0)
            else:
                ax.plot(i, h * 1.0006, marker='v', markersize=marker_size,
                        color=RED, alpha=alpha, zorder=5, markeredgewidth=0)

        # 캔들 아래 예측 결과 텍스트
        # 실제 결과 있으면 맞/틀림 표시, 없으면 예측만 표시
        actual = r.get('actual_next')
        actual_ret = r.get('actual_ret')
        if pred != 'NEUTRAL' and conf >= 55:
            if actual is not None:
                # 실제 결과 있음 → 맞/틀림 판정
                is_correct = (
                    (pred == 'BULLISH' and actual == 'BULLISH') or
                    (pred == 'BEARISH' and actual == 'BEARISH')
                )
                txt_color = GREEN if is_correct else RED
                mark = '✓' if is_correct else '✗'
                ret_str = f"{actual_ret:+.1f}%" if actual_ret is not None else ''
                ax.text(i, l * 0.9988, f"{mark}{ret_str}",
                        ha='center', va='top', fontsize=5.5,
                        color=txt_color, alpha=0.85, zorder=6)
            else:
                # 마지막 봉 — 예측만 표시 (NOW 강조)
                txt = '▲NOW' if pred == 'BULLISH' else '▼NOW'
                clr = GREEN  if pred == 'BULLISH' else RED
                ax.text(i, l * 0.9988, txt,
                        ha='center', va='top', fontsize=6,
                        color=clr, fontweight='bold', alpha=0.95, zorder=6)

    # 볼린저 밴드
    if bb_upper is not None and bb_lower is not None:
        x = np.arange(len(results))
        ax.plot(x, bb_upper, color=BLUE, linewidth=0.7, alpha=0.5, linestyle='--', zorder=4)
        ax.plot(x, bb_lower, color=BLUE, linewidth=0.7, alpha=0.5, linestyle='--', zorder=4)
        ax.fill_between(x, bb_upper, bb_lower, alpha=0.04, color=BLUE, zorder=1)


# ── 메인 시각화 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--bars',   type=int, default=50)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n예측 시각화: {ticker}  ({args.bars}봉)")

    results = fetch_and_predict(ticker, args.bars)
    latest  = results[-1]
    n       = len(results)

    pred_color = GREEN if latest['prediction'] == 'BULLISH' else \
                 (RED  if latest['prediction'] == 'BEARISH' else GOLD)
    icon = '▲' if latest['prediction'] == 'BULLISH' else \
           ('▼' if latest['prediction'] == 'BEARISH' else '─')

    # 볼린저 밴드 계산
    closes = pd.Series([r['close'] for r in results])
    if len(closes) >= 20:
        ma20     = closes.rolling(20).mean()
        std20    = closes.rolling(20).std()
        bb_upper = (ma20 + 2 * std20).values
        bb_lower = (ma20 - 2 * std20).values
    else:
        bb_upper = bb_lower = None

    # 최근 N봉 실제 승률 계산 (실제 결과 있는 봉만)
    verifiable = [r for r in results
                  if r['prediction'] in ('BULLISH', 'BEARISH')
                  and r['confidence'] >= 55
                  and r.get('actual_next') is not None]
    correct = [r for r in verifiable
               if (r['prediction'] == 'BULLISH' and r['actual_next'] == 'BULLISH') or
                  (r['prediction'] == 'BEARISH' and r['actual_next'] == 'BEARISH')]
    win_rate     = len(correct) / len(verifiable) * 100 if verifiable else None
    win_rate_cnt = f"{len(correct)}/{len(verifiable)}"

    # ── 레이아웃 ────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 17), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(
        5, 4, figure=fig,
        height_ratios=[0.55, 3.5, 0.45, 1.0, 1.0],
        width_ratios=[3, 3, 3, 1.4],
        hspace=0.06, wspace=0.04,
        top=0.96, bottom=0.05, left=0.05, right=0.98
    )

    ax_head   = fig.add_subplot(gs[0, :3])
    ax_candle = fig.add_subplot(gs[1, :3])
    ax_label  = fig.add_subplot(gs[2, :3])   # ← 예측 라벨 행
    ax_vol    = fig.add_subplot(gs[3, :3])
    ax_rsi    = fig.add_subplot(gs[4, :2])
    ax_macd   = fig.add_subplot(gs[4, 2])
    ax_panel  = fig.add_subplot(gs[:, 3])

    def style_ax(ax, bottom_ticks=False):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=WHITE2, labelsize=8, length=3)
        ax.tick_params(axis='x', colors=WHITE2 if bottom_ticks else DARK)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(0.8)

    for ax in [ax_head, ax_candle, ax_vol, ax_panel]:
        style_ax(ax)
    style_ax(ax_label)
    style_ax(ax_rsi)
    style_ax(ax_macd)
    ax_rsi.tick_params(axis='x', colors=WHITE2)
    ax_macd.tick_params(axis='x', colors=WHITE2)

    x_idx     = np.arange(n)
    times     = [r['timestamp'].strftime('%H:%M') for r in results]
    tick_step = max(1, n // 10)

    # ① 헤더
    ax_head.set_facecolor(CARD2)
    ax_head.axis('off')

    # 종목명 + 가격
    ax_head.text(0.01, 0.72, ticker,
                 transform=ax_head.transAxes, fontsize=18,
                 color=WHITE, fontweight='bold', va='center')
    ax_head.text(0.10, 0.72, f"${latest['close']:.2f}",
                 transform=ax_head.transAxes, fontsize=16,
                 color=WHITE, va='center')
    ax_head.text(0.10, 0.22, latest['timestamp'].strftime('%Y-%m-%d  %H:%M EST'),
                 transform=ax_head.transAxes, fontsize=9,
                 color=GRAY, va='center')

    # 예측 결과 박스
    box_x = 0.30
    ax_head.text(box_x, 0.72,
                 f"{icon}  {latest['pred_ko']}",
                 transform=ax_head.transAxes, fontsize=17,
                 color=pred_color, fontweight='bold', va='center')
    ax_head.text(box_x, 0.22,
                 latest['strength'],
                 transform=ax_head.transAxes, fontsize=9,
                 color=pred_color, alpha=0.8, va='center')

    # 신뢰도 바
    conf     = latest['confidence']
    bar_w    = 0.22
    bar_x    = 0.50
    bar_y    = 0.35
    bar_h    = 0.28
    conf_clr = pred_color
    ax_head.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0.01", linewidth=0.5,
        facecolor=GRAY2, edgecolor=BORDER,
        transform=ax_head.transAxes, clip_on=False
    ))
    ax_head.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w * conf / 100, bar_h,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=conf_clr, alpha=0.85,
        transform=ax_head.transAxes, clip_on=False
    ))
    ax_head.text(bar_x + bar_w / 2, bar_y + bar_h / 2,
                 f"신뢰도  {conf:.0f}%",
                 transform=ax_head.transAxes, fontsize=10,
                 color=WHITE, fontweight='bold', va='center', ha='center')

    # 신호 분포 (우측 헤더)
    buy_cnt  = sum(1 for r in results if r['prediction'] == 'BULLISH')
    sell_cnt = sum(1 for r in results if r['prediction'] == 'BEARISH')
    hold_cnt = sum(1 for r in results if r['prediction'] == 'NEUTRAL')
    stat_x   = 0.78
    for i, (label, cnt, clr) in enumerate([
        (f'▲ 매수  {buy_cnt}',  buy_cnt,  GREEN),
        (f'▼ 매도  {sell_cnt}', sell_cnt, RED),
        (f'─ 관망  {hold_cnt}', hold_cnt, GOLD),
    ]):
        ax_head.text(stat_x + i * 0.075, 0.5, label,
                     transform=ax_head.transAxes, fontsize=9,
                     color=clr, va='center', ha='center', fontweight='bold')

    # ② 캔들차트
    draw_candles(ax_candle, results, bb_upper, bb_lower)

    vwaps  = [r['vwap'] for r in results]
    ax_candle.plot(x_idx, vwaps, color=GOLD, linewidth=1.0,
                   linestyle='--', alpha=0.6, label='VWAP', zorder=4)
    if len(closes) >= 20:
        ax_candle.plot(x_idx, ma20.values, color=BLUE, linewidth=0.9,
                       alpha=0.5, label='MA20', zorder=4)

    ax_candle.set_xlim(-0.8, n - 0.2)
    ax_candle.set_xticks(x_idx[::tick_step])
    ax_candle.set_xticklabels([])
    ax_candle.set_ylabel('가격 ($)', color=WHITE2, fontsize=8, labelpad=6)
    ax_candle.set_title(
        f'  5분봉  |  ▲▼ 신호 크기 = 신뢰도  |  볼린저 밴드 (20, 2σ)',
        color=WHITE2, fontsize=9, pad=6, loc='left'
    )
    ax_candle.legend(
        facecolor=CARD2, edgecolor=BORDER, labelcolor=WHITE2,
        fontsize=8, loc='upper left', framealpha=0.8
    )
    ax_candle.yaxis.set_tick_params(labelcolor=WHITE2)

    # 현재 가격 수평선
    ax_candle.axhline(latest['close'], color=WHITE, linewidth=0.6,
                      linestyle=':', alpha=0.4, zorder=3)
    price_range = ax_candle.get_ylim()
    ax_candle.text(n - 0.2, latest['close'],
                   f"  ${latest['close']:.2f}",
                   color=WHITE, fontsize=8, va='center', alpha=0.7)

    # 예상 종가 수평선 (회귀 예측 있을 때만)
    if latest.get('pred_close') is not None:
        pred_clr = GREEN if latest['pred_change'] > 0 else RED
        ax_candle.axhline(latest['pred_close'], color=pred_clr,
                          linewidth=0.8, linestyle='-.', alpha=0.6, zorder=3)
        ax_candle.text(n - 0.2, latest['pred_close'],
                       f"  ${latest['pred_close']:.2f} ({latest['pred_change']:+.2f}%)",
                       color=pred_clr, fontsize=7.5, va='center', alpha=0.85)

    # ③ 예측 라벨 행 ─────────────────────────────────────────
    ax_label.set_facecolor('#0d1520')
    ax_label.set_xlim(-0.8, n - 0.2)
    ax_label.set_ylim(0, 1)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    ax_label.set_ylabel('예측', color=GRAY, fontsize=7, labelpad=6)

    for i, r in enumerate(results):
        pred = r['prediction']
        conf = r['confidence']

        if pred == 'BULLISH':
            icon_lbl = '▲'
            pct_lbl  = f"{r['prob_bull']:.0f}%"
            clr      = GREEN
        elif pred == 'BEARISH':
            icon_lbl = '▼'
            pct_lbl  = f"{r['prob_bear']:.0f}%"
            clr      = RED
        else:
            icon_lbl = '─'
            pct_lbl  = f"{r['prob_neut']:.0f}%"
            clr      = GRAY

        # 신뢰도 55% 이상만 색상 강조, 미만은 흐리게
        alpha = 0.9 if conf >= 55 else 0.3

        # 마지막 봉 (현재 예측) 배경 강조
        if i == n - 1:
            ax_label.axvspan(i - 0.5, i + 0.5,
                             alpha=0.15, color=clr, zorder=1)

        ax_label.text(i, 0.65, icon_lbl,
                      ha='center', va='center', fontsize=7,
                      color=clr, alpha=alpha, fontweight='bold', zorder=3)
        ax_label.text(i, 0.25, pct_lbl,
                      ha='center', va='center', fontsize=6,
                      color=clr, alpha=alpha, zorder=3)
    vols       = [r['volume'] for r in results]
    vol_colors = []
    for r in results:
        alpha = 0.5 + (r['confidence'] - 50) / 100 if r['prediction'] != 'NEUTRAL' else 0.4
        vol_colors.append(GREEN if r['close'] >= r['open'] else RED)

    ax_vol.bar(x_idx, vols, color=vol_colors, alpha=0.65, width=0.65, zorder=3)
    vol_mean = np.mean(vols)
    ax_vol.axhline(vol_mean, color=GOLD, linewidth=0.8, linestyle='--',
                   alpha=0.6, label=f'평균', zorder=4)
    ax_vol.set_xlim(-0.8, n - 0.2)
    ax_vol.set_xticks(x_idx[::tick_step])
    ax_vol.set_xticklabels([])
    ax_vol.set_ylabel('거래량', color=WHITE2, fontsize=8, labelpad=6)
    ax_vol.yaxis.set_tick_params(labelcolor=WHITE2)
    ax_vol.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=WHITE2,
                  fontsize=7, loc='upper left', framealpha=0.8)

    # ④ RSI
    rsi_vals = [r['rsi14'] for r in results]
    rsi_colors = []
    for v in rsi_vals:
        if v >= 70:   rsi_colors.append(RED)
        elif v <= 30: rsi_colors.append(GREEN)
        else:         rsi_colors.append(BLUE)

    ax_rsi.plot(x_idx, rsi_vals, color=BLUE, linewidth=1.3, zorder=4)
    ax_rsi.fill_between(x_idx, rsi_vals, 70,
                         where=[v > 70 for v in rsi_vals],
                         alpha=0.2, color=RED, zorder=2)
    ax_rsi.fill_between(x_idx, rsi_vals, 30,
                         where=[v < 30 for v in rsi_vals],
                         alpha=0.2, color=GREEN, zorder=2)
    ax_rsi.axhline(70, color=RED,   linewidth=0.6, linestyle='--', alpha=0.5)
    ax_rsi.axhline(50, color=GRAY,  linewidth=0.4, linestyle=':',  alpha=0.3)
    ax_rsi.axhline(30, color=GREEN, linewidth=0.6, linestyle='--', alpha=0.5)
    ax_rsi.set_xlim(-0.8, n - 0.2)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_xticks(x_idx[::tick_step])
    ax_rsi.set_xticklabels([times[i] for i in x_idx[::tick_step]],
                             color=WHITE2, fontsize=7, rotation=30)
    ax_rsi.set_ylabel('RSI(14)', color=WHITE2, fontsize=8, labelpad=6)
    ax_rsi.yaxis.set_tick_params(labelcolor=WHITE2)
    ax_rsi.text(0.01, 0.88, '과매수 70', transform=ax_rsi.transAxes,
                color=RED, fontsize=7, alpha=0.7)
    ax_rsi.text(0.01, 0.08, '과매도 30', transform=ax_rsi.transAxes,
                color=GREEN, fontsize=7, alpha=0.7)

    # ⑤ MACD
    macd_vals   = [r['macd_hist'] for r in results]
    macd_colors = [GREEN if v >= 0 else RED for v in macd_vals]
    ax_macd.bar(x_idx, macd_vals, color=macd_colors, alpha=0.75, width=0.65, zorder=3)
    ax_macd.axhline(0, color=GRAY2, linewidth=0.8, zorder=4)
    ax_macd.set_xlim(-0.8, n - 0.2)
    ax_macd.set_xticks(x_idx[::tick_step])
    ax_macd.set_xticklabels([times[i] for i in x_idx[::tick_step]],
                              color=WHITE2, fontsize=7, rotation=30)
    ax_macd.set_ylabel('MACD Hist', color=WHITE2, fontsize=8, labelpad=6)
    ax_macd.yaxis.set_tick_params(labelcolor=WHITE2)

    # ⑥ 우측 예측 패널 ──────────────────────────────────────
    ax_panel.set_facecolor(CARD2)
    ax_panel.axis('off')

    # 제목
    ax_panel.text(0.5, 0.975, '예측 결과',
                  transform=ax_panel.transAxes, fontsize=11,
                  color=WHITE, ha='center', va='top', fontweight='bold')

    # 예측 신호 크게 표시
    ax_panel.text(0.5, 0.895,
                  f"{icon} {latest['pred_ko']}",
                  transform=ax_panel.transAxes, fontsize=22,
                  color=pred_color, ha='center', va='center', fontweight='bold')
    ax_panel.text(0.5, 0.845,
                  latest['strength'],
                  transform=ax_panel.transAxes, fontsize=9,
                  color=pred_color, ha='center', va='center', alpha=0.8)

    # 구분선
    ax_panel.plot([0.05, 0.95], [0.825, 0.825], color=BORDER, linewidth=0.8,
                  transform=ax_panel.transAxes)

    # 확률 게이지
    def gauge(y_base, label, pct, color):
        bar_h = 0.032
        ax_panel.text(0.06, y_base + bar_h + 0.01, label,
                      transform=ax_panel.transAxes, fontsize=8,
                      color=WHITE2, va='bottom')
        ax_panel.text(0.94, y_base + bar_h + 0.01, f'{pct:.1f}%',
                      transform=ax_panel.transAxes, fontsize=8,
                      color=color, va='bottom', ha='right', fontweight='bold')
        # 배경
        ax_panel.add_patch(FancyBboxPatch(
            (0.06, y_base), 0.88, bar_h,
            boxstyle="round,pad=0.003", linewidth=0,
            facecolor=GRAY2, transform=ax_panel.transAxes
        ))
        # 채움
        if pct > 0:
            fill_w = max(0.88 * pct / 100, 0.02)
            ax_panel.add_patch(FancyBboxPatch(
                (0.06, y_base), fill_w, bar_h,
                boxstyle="round,pad=0.003", linewidth=0,
                facecolor=color, alpha=0.85,
                transform=ax_panel.transAxes
            ))

    gauge(0.756, '▲ 매수 확률', latest['prob_bull'], GREEN)
    gauge(0.700, '▼ 매도 확률', latest['prob_bear'], RED)
    gauge(0.644, '─ 관망 확률', latest['prob_neut'], GOLD)

    # 구분선
    ax_panel.plot([0.05, 0.95], [0.625, 0.625], color=BORDER, linewidth=0.8,
                  transform=ax_panel.transAxes)

    # 수치 정보
    def info_row(y, label, value, color=WHITE2):
        ax_panel.text(0.06, y, label,
                      transform=ax_panel.transAxes, fontsize=8.5,
                      color=GRAY, va='center')
        ax_panel.text(0.94, y, value,
                      transform=ax_panel.transAxes, fontsize=9,
                      color=color, va='center', ha='right', fontweight='bold')

    info_row(0.590, '현재 종가',   f"${latest['close']:.2f}", WHITE)

    # 회귀 예측 (있을 때만)
    if latest.get('pred_change') is not None:
        pred_clr = GREEN if latest['pred_change'] > 0 else RED
        info_row(0.540, '예상 변동폭', f"{latest['pred_change']:+.3f}%", pred_clr)
        info_row(0.490, '예상 종가',   f"${latest['pred_close']:.2f}",   pred_clr)
        # 구분선
        ax_panel.plot([0.05, 0.95], [0.468, 0.468], color=BORDER, linewidth=0.6,
                      transform=ax_panel.transAxes)
        info_row(0.435, 'RSI (14)',   f"{latest['rsi14']:.1f}",
                 RED if latest['rsi14'] > 70 else (GREEN if latest['rsi14'] < 30 else BLUE))
        info_row(0.390, 'VWAP 이격',
                 f"{latest['dist_vwap']:+.2f}%",
                 GREEN if latest['dist_vwap'] > 0 else RED)
    else:
        info_row(0.535, 'RSI (14)',   f"{latest['rsi14']:.1f}",
                 RED if latest['rsi14'] > 70 else (GREEN if latest['rsi14'] < 30 else BLUE))
        info_row(0.480, 'VWAP 이격',
                 f"{latest['dist_vwap']:+.2f}%",
                 GREEN if latest['dist_vwap'] > 0 else RED)

    # BB 위치
    if bb_upper is not None and bb_lower is not None:
        last_upper = bb_upper[-1]
        last_lower = bb_lower[-1]
        last_close = latest['close']
        bb_range   = last_upper - last_lower
        bb_pos     = (last_close - last_lower) / bb_range * 100 if bb_range > 0 else 50
        bb_color   = RED if bb_pos > 85 else (GREEN if bb_pos < 15 else WHITE2)
        y_bb = 0.345 if latest.get('pred_change') is not None else 0.425
        info_row(y_bb, 'BB 위치', f"{bb_pos:.0f}%", bb_color)

    # 구분선
    ax_panel.plot([0.05, 0.95], [0.385, 0.385], color=BORDER, linewidth=0.8,
                  transform=ax_panel.transAxes)

    # 신호 통계
    ax_panel.text(0.5, 0.345, f'최근 {n}봉 신호',
                  transform=ax_panel.transAxes, fontsize=8,
                  color=GRAY, ha='center', va='center')

    for i, (lbl, cnt, clr) in enumerate([
        ('매수', buy_cnt, GREEN),
        ('매도', sell_cnt, RED),
        ('관망', hold_cnt, GOLD),
    ]):
        bx = 0.15 + i * 0.285
        ax_panel.text(bx, 0.275, str(cnt),
                      transform=ax_panel.transAxes, fontsize=14,
                      color=clr, ha='center', va='center', fontweight='bold')
        ax_panel.text(bx, 0.225, lbl,
                      transform=ax_panel.transAxes, fontsize=7.5,
                      color=GRAY, ha='center', va='center')

    # 구분선
    ax_panel.plot([0.05, 0.95], [0.200, 0.200], color=BORDER, linewidth=0.8,
                  transform=ax_panel.transAxes)

    # 예측 적중률 (신뢰도 55%+)
    ax_panel.text(0.5, 0.178, '예측 적중률  (신뢰도 55%+)',
                  transform=ax_panel.transAxes, fontsize=8,
                  color=GRAY, ha='center', va='center')

    if win_rate is not None and len(verifiable) >= 3:
        wr_color = GREEN if win_rate >= 55 else (GOLD if win_rate >= 45 else RED)
        ax_panel.text(0.5, 0.118,
                      f"{win_rate:.0f}%",
                      transform=ax_panel.transAxes, fontsize=22,
                      color=wr_color, ha='center', va='center', fontweight='bold')
        ax_panel.text(0.5, 0.065,
                      f"{win_rate_cnt}  맞음/전체",
                      transform=ax_panel.transAxes, fontsize=8,
                      color=GRAY, ha='center', va='center')
    else:
        ax_panel.text(0.5, 0.100,
                      '데이터 부족',
                      transform=ax_panel.transAxes, fontsize=10,
                      color=GRAY, ha='center', va='center')

    ax_panel.text(0.5, 0.022,
                  latest['timestamp'].strftime('%Y-%m-%d'),
                  transform=ax_panel.transAxes, fontsize=7.5,
                  color=GRAY, ha='center', va='center')

    # ── 저장 ────────────────────────────────────────────────
    save_path = os.path.join(DESKTOP, f'{ticker}_predict_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK, edgecolor='none')
    plt.close()
    print(f"\n저장 완료: {save_path}")


if __name__ == '__main__':
    main()
