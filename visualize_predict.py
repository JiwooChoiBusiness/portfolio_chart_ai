"""
예측 결과 시각화
================
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
DARK      = '#0d1117'
CARD      = '#161b22'
GREEN     = '#2ea043'
RED       = '#f85149'
BLUE      = '#58a6ff'
GOLD      = '#e3b341'
GRAY      = '#8b949e'
WHITE     = '#e6edf3'
BULL_BODY = '#26a641'
BEAR_BODY = '#f85149'

# ── 경로 ────────────────────────────────────────────────────
DESKTOP   = os.path.join(os.path.expanduser('~'), 'Desktop')
BASE_DIR  = os.path.join(DESKTOP, 'chart_ml')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# ── 데이터 수집 & 예측 ──────────────────────────────────────
def fetch_and_predict(ticker: str, n_bars: int = 30):
    import sys
    sys.path.insert(0, BASE_DIR)
    from data_pipeline_5m import (
        add_candle_features, add_pattern_features, add_ma_features,
        add_vwap_features, add_session_features, add_momentum_features,
        add_volume_features, add_volatility_features,
        add_support_resistance_features, add_lag_features
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, 'chart_model.json'))
    with open(os.path.join(MODEL_DIR, 'feature_list.txt')) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    with open(os.path.join(MODEL_DIR, 'label_classes.json')) as f:
        label_classes = json.load(f)

    n_predict = min(n_bars, len(df))
    X = pd.DataFrame(index=df.index[-n_predict:])
    for col in feature_cols:
        X[col] = df[col].iloc[-n_predict:].values if col in df.columns else 0.0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    proba     = model.predict_proba(X.values)
    preds_enc = model.predict(X.values)

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
        results.append({
            'timestamp':  X.index[i],
            'open':       float(row['open']),
            'high':       float(row['high']),
            'low':        float(row['low']),
            'close':      float(row['close']),
            'volume':     float(row['volume']),
            'rsi14':      float(row['rsi14'])     if 'rsi14'     in df.columns else 50.0,
            'macd_hist':  float(row['macd_hist']) if 'macd_hist' in df.columns else 0.0,
            'vwap':       float(row['vwap'])      if 'vwap'      in df.columns else float(row['close']),
            'dist_vwap':  float(row['dist_vwap']) if 'dist_vwap' in df.columns else 0.0,
            'prediction': direction,
            'pred_ko':    PRED_KO[direction],
            'strength':   STRENGTH_KO.get(strength, strength),
            'prob_bull':  round(prob_bull * 100, 1),
            'prob_bear':  round(prob_bear * 100, 1),
            'prob_neut':  round(prob_neut * 100, 1),
            'confidence': round(max_prob * 100, 1),
        })

    return results


# ── 캔들 그리기 ─────────────────────────────────────────────
def draw_candles(ax, results):
    for i, r in enumerate(results):
        o, h, l, c = r['open'], r['high'], r['low'], r['close']
        color  = BULL_BODY if c >= o else BEAR_BODY
        body_b = min(o, c)
        body_h = abs(c - o) if abs(c - o) > 0 else (h - l) * 0.01

        ax.plot([i, i], [l, h], color=color, linewidth=0.8, alpha=0.9)
        ax.bar(i, body_h, bottom=body_b, width=0.6,
               color=color, alpha=0.85, edgecolor=color, linewidth=0.3)

        pred = r['prediction']
        conf = r['confidence']
        if pred == 'BULLISH' and conf >= 55:
            ax.annotate('▲', xy=(i, l * 0.9995), fontsize=6,
                        color=GREEN, ha='center', va='top', alpha=0.9)
        elif pred == 'BEARISH' and conf >= 55:
            ax.annotate('▼', xy=(i, h * 1.0005), fontsize=6,
                        color=RED, ha='center', va='bottom', alpha=0.9)


# ── 메인 시각화 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--bars',   type=int, default=30)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n예측 시각화: {ticker}  ({args.bars}봉)")

    results = fetch_and_predict(ticker, args.bars)
    latest  = results[-1]
    n       = len(results)

    pred_color = GREEN if latest['prediction'] == 'BULLISH' else \
                 (RED  if latest['prediction'] == 'BEARISH' else GOLD)
    icon = '▲' if latest['prediction'] == 'BULLISH' else \
           ('▼' if latest['prediction'] == 'BEARISH' else '−')

    # ── 레이아웃 ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=DARK)
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            height_ratios=[0.5, 3, 1.2, 1.2, 1.2],
                            hspace=0.12, wspace=0.3,
                            top=0.95, bottom=0.04, left=0.06, right=0.97)

    ax_head   = fig.add_subplot(gs[0, :])
    ax_candle = fig.add_subplot(gs[1, :])
    ax_vol    = fig.add_subplot(gs[2, :])
    ax_rsi    = fig.add_subplot(gs[3, :2])
    ax_macd   = fig.add_subplot(gs[4, :2])
    ax_panel  = fig.add_subplot(gs[3:, 2])

    for ax in [ax_head, ax_candle, ax_vol, ax_rsi, ax_macd, ax_panel]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=WHITE, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')

    x_idx     = np.arange(n)
    times     = [r['timestamp'].strftime('%H:%M') for r in results]
    tick_step = max(1, n // 8)

    # ① 헤더
    ax_head.axis('off')
    ax_head.text(0.02, 0.55,
                 f"{ticker}   ${latest['close']:.2f}   |   최신: {latest['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                 transform=ax_head.transAxes, fontsize=14, color=WHITE,
                 fontweight='bold', va='center')
    ax_head.text(0.72, 0.55,
                 f"{icon}  {latest['pred_ko']}  ({latest['strength']})   신뢰도 {latest['confidence']:.0f}%",
                 transform=ax_head.transAxes, fontsize=13, color=pred_color,
                 fontweight='bold', va='center')

    # ② 캔들차트
    draw_candles(ax_candle, results)

    vwaps  = [r['vwap'] for r in results]
    closes = pd.Series([r['close'] for r in results])
    ax_candle.plot(x_idx, vwaps, color=GOLD, linewidth=1.0, linestyle='--', alpha=0.7, label='VWAP')
    if len(closes) >= 20:
        ma20 = closes.rolling(20).mean()
        ax_candle.plot(x_idx, ma20, color=BLUE, linewidth=0.9, alpha=0.6, label='MA20')

    ax_candle.set_xlim(-0.5, n - 0.5)
    ax_candle.set_xticks(x_idx[::tick_step])
    ax_candle.set_xticklabels([])
    ax_candle.set_ylabel('가격 ($)', color=GRAY, fontsize=9)
    ax_candle.set_title(f'{ticker}  5분봉 캔들차트  (▲=매수신호  ▼=매도신호, 신뢰도 55% 이상)',
                        color=WHITE, fontsize=11, pad=8)
    ax_candle.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=WHITE,
                     fontsize=8, loc='upper left')

    for i, r in enumerate(results):
        if r['prediction'] == 'BULLISH' and r['confidence'] >= 55:
            ax_candle.axvspan(i-0.5, i+0.5, alpha=0.06, color=GREEN)
        elif r['prediction'] == 'BEARISH' and r['confidence'] >= 55:
            ax_candle.axvspan(i-0.5, i+0.5, alpha=0.06, color=RED)

    # ③ 거래량
    vols        = [r['volume'] for r in results]
    vol_colors  = [GREEN if r['close'] >= r['open'] else RED for r in results]
    vol_mean    = np.mean(vols)
    ax_vol.bar(x_idx, vols, color=vol_colors, alpha=0.7, width=0.6)
    ax_vol.axhline(vol_mean, color=GOLD, linewidth=0.8, linestyle='--',
                   alpha=0.7, label=f'평균 {vol_mean:,.0f}')
    ax_vol.set_xlim(-0.5, n - 0.5)
    ax_vol.set_xticks(x_idx[::tick_step])
    ax_vol.set_xticklabels([])
    ax_vol.set_ylabel('거래량', color=GRAY, fontsize=8)
    ax_vol.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=WHITE,
                  fontsize=7, loc='upper left')

    # ④ RSI
    rsi_vals = [r['rsi14'] for r in results]
    ax_rsi.plot(x_idx, rsi_vals, color=BLUE, linewidth=1.2)
    ax_rsi.axhline(70, color=RED,   linewidth=0.7, linestyle='--', alpha=0.6)
    ax_rsi.axhline(30, color=GREEN, linewidth=0.7, linestyle='--', alpha=0.6)
    ax_rsi.axhline(50, color=GRAY,  linewidth=0.5, linestyle=':',  alpha=0.4)
    ax_rsi.fill_between(x_idx, rsi_vals, 70, where=[v > 70 for v in rsi_vals],
                         alpha=0.15, color=RED)
    ax_rsi.fill_between(x_idx, rsi_vals, 30, where=[v < 30 for v in rsi_vals],
                         alpha=0.15, color=GREEN)
    ax_rsi.set_xlim(-0.5, n - 0.5)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_xticks(x_idx[::tick_step])
    ax_rsi.set_xticklabels([])
    ax_rsi.set_ylabel('RSI(14)', color=GRAY, fontsize=8)
    ax_rsi.text(n-1, 72, '과매수', color=RED,   fontsize=7, ha='right')
    ax_rsi.text(n-1, 32, '과매도', color=GREEN, fontsize=7, ha='right')

    # ⑤ MACD 히스토그램
    macd_vals   = [r['macd_hist'] for r in results]
    macd_colors = [GREEN if v >= 0 else RED for v in macd_vals]
    ax_macd.bar(x_idx, macd_vals, color=macd_colors, alpha=0.75, width=0.6)
    ax_macd.axhline(0, color=GRAY, linewidth=0.7)
    ax_macd.set_xlim(-0.5, n - 0.5)
    ax_macd.set_xticks(x_idx[::tick_step])
    ax_macd.set_xticklabels([times[i] for i in x_idx[::tick_step]],
                              color=WHITE, fontsize=7, rotation=30)
    ax_macd.set_ylabel('MACD Hist', color=GRAY, fontsize=8)

    # ⑥ 우측 예측 패널
    ax_panel.axis('off')

    def panel_text(ax, y, label, value, color=WHITE):
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=9, color=GRAY, va='center')
        ax.text(0.95, y, value, transform=ax.transAxes,
                fontsize=10, color=color, va='center', ha='right', fontweight='bold')

    ax_panel.text(0.5, 0.97, '최신 봉 예측',
                  transform=ax_panel.transAxes, fontsize=11, color=WHITE,
                  ha='center', va='top', fontweight='bold')

    def gauge(ax, y_base, label, pct, color, bar_h=0.045):
        ax.text(0.05, y_base + bar_h/2 + 0.01, label,
                transform=ax.transAxes, fontsize=8, color=GRAY, va='bottom')
        ax.text(0.95, y_base + bar_h/2 + 0.01, f'{pct:.1f}%',
                transform=ax.transAxes, fontsize=8, color=color,
                va='bottom', ha='right')
        bg = FancyBboxPatch((0.05, y_base), 0.9, bar_h,
                             boxstyle="round,pad=0.005", linewidth=0,
                             facecolor='#21262d', transform=ax.transAxes)
        ax.add_patch(bg)
        if pct > 0:
            fill = FancyBboxPatch((0.05, y_base), 0.9 * pct/100, bar_h,
                                   boxstyle="round,pad=0.005", linewidth=0,
                                   facecolor=color, alpha=0.8,
                                   transform=ax.transAxes)
            ax.add_patch(fill)

    gauge(ax_panel, 0.82, '매수 확률', latest['prob_bull'], GREEN)
    gauge(ax_panel, 0.73, '매도 확률', latest['prob_bear'], RED)
    gauge(ax_panel, 0.64, '관망 확률', latest['prob_neut'], GOLD)

    panel_text(ax_panel, 0.57, '예측 신호',  latest['pred_ko'],       pred_color)
    panel_text(ax_panel, 0.50, '상승/하락세', latest['strength'],      WHITE)
    panel_text(ax_panel, 0.43, '신뢰도',     f"{latest['confidence']:.1f}%", pred_color)
    panel_text(ax_panel, 0.36, '종가',       f"${latest['close']:.2f}", WHITE)
    panel_text(ax_panel, 0.29, 'RSI(14)',    f"{latest['rsi14']:.1f}", BLUE)
    panel_text(ax_panel, 0.22, 'VWAP 이격',
               f"{latest['dist_vwap']:+.2f}%",
               GREEN if latest['dist_vwap'] > 0 else RED)

    buy_cnt  = sum(1 for r in results if r['prediction'] == 'BULLISH')
    sell_cnt = sum(1 for r in results if r['prediction'] == 'BEARISH')
    hold_cnt = sum(1 for r in results if r['prediction'] == 'NEUTRAL')
    ax_panel.text(0.5, 0.13,
                  f"최근 {n}봉 신호 분포\n매수 {buy_cnt}  매도 {sell_cnt}  관망 {hold_cnt}",
                  transform=ax_panel.transAxes, fontsize=9, color=GRAY,
                  ha='center', va='center', linespacing=1.6)

    # ── 저장 ────────────────────────────────────────────────
    save_path = os.path.join(DESKTOP, f'{ticker}_predict_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"\n저장 완료: {save_path}")


if __name__ == '__main__':
    main()
