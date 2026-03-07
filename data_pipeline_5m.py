"""
5분봉 차트 ML 파이프라인 - 데이터 수집 & 피처 생성
=====================================================
사용법:
  pip install yfinance pandas numpy scikit-learn --break-system-packages

  # 단일 종목
  python data_pipeline_5m.py --ticker AAPL

  # 여러 종목 (범용 모델용 권장)
  python data_pipeline_5m.py --ticker AAPL MSFT NVDA TSLA META GOOGL AMZN

출력:
  output/AAPL_5m_features.csv
  output/combined_5m_features.csv  (여러 종목 합본 → train_model.py 입력)

Yahoo Finance 5분봉 제한:
  - 최대 60일치만 제공
  - 60일 × 6.5시간 × 12봉 = 약 4,680개/종목
  - 7개 종목 합산 시 약 32,000개 → 학습에 충분
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────
# 1. 데이터 수집
# ─────────────────────────────────────────────────────────

def fetch_5m_ohlcv(ticker: str) -> pd.DataFrame:
    """Yahoo Finance 5분봉 수집 (최대 60일)"""
    print(f"  [{ticker}] 5분봉 수집 중...")
    df = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"데이터를 가져오지 못했습니다: {ticker}")

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    # 장외 시간 제거 (09:30 ~ 16:00 EST만 유지)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York')
    df = df.iloc[df.index.indexer_between_time('09:30', '15:55')]

    df.dropna(inplace=True)
    print(f"  [{ticker}] {len(df)}개 5분봉 수집 완료 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    return df


# ─────────────────────────────────────────────────────────
# 2. 캔들 피처
# ─────────────────────────────────────────────────────────

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df['open'], df['high'], df['low'], df['close']

    df['body']             = c - o
    df['body_abs']         = df['body'].abs()
    df['upper_wick']       = h - np.maximum(o, c)
    df['lower_wick']       = np.minimum(o, c) - l
    df['candle_range']     = h - l

    rng = df['candle_range'].replace(0, np.nan)
    df['body_ratio']       = df['body_abs'] / rng
    df['upper_wick_ratio'] = df['upper_wick'] / rng
    df['lower_wick_ratio'] = df['lower_wick'] / rng
    df['is_bullish']       = (c > o).astype(int)

    # 갭
    df['gap_pct'] = (o - c.shift(1)) / c.shift(1) * 100

    # 연속 상승/하락
    streaks = []
    streak = 0
    for bull in df['is_bullish']:
        streak = max(streak + 1, 1) if bull == 1 else min(streak - 1, -1)
        streaks.append(streak)
    df['streak'] = streaks

    return df


# ─────────────────────────────────────────────────────────
# 3. 캔들 패턴
# ─────────────────────────────────────────────────────────

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    body = df['body_abs']
    rng  = df['candle_range'].replace(0, np.nan)

    df['pat_doji']          = (body / rng < 0.1).astype(int)
    df['pat_hammer']        = ((df['lower_wick'] > body * 2) & (df['upper_wick'] < body * 0.5) & (df['is_bullish'] == 1)).astype(int)
    df['pat_shooting_star'] = ((df['upper_wick'] > body * 2) & (df['lower_wick'] < body * 0.3) & (body / rng < 0.35)).astype(int)
    df['pat_bullish_engulf']= ((df['is_bullish'] == 1) & (df['is_bullish'].shift(1) == 0) & (o < c.shift(1)) & (c > o.shift(1))).astype(int)
    df['pat_bearish_engulf']= ((df['is_bullish'] == 0) & (df['is_bullish'].shift(1) == 1) & (o > c.shift(1)) & (c < o.shift(1))).astype(int)
    df['pat_marubozu_bull'] = ((body / rng > 0.8) & (df['is_bullish'] == 1)).astype(int)
    df['pat_marubozu_bear'] = ((body / rng > 0.8) & (df['is_bullish'] == 0)).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 4. 이동평균 (분봉 적합한 단위)
# ─────────────────────────────────────────────────────────

def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    5분봉 기준 이동평균
      9봉  = 약 45분 (단기)
      20봉 = 약 1.7시간
      50봉 = 약 4시간 (반나절)
      200봉 = 약 1.7일 (단기 추세)
    """
    c = df['close']

    for w in [9, 20, 50, 200]:
        df[f'ma{w}']      = c.rolling(w).mean()
        df[f'dist_ma{w}'] = (c - df[f'ma{w}']) / df[f'ma{w}'] * 100

    df['ma9_above_ma20']   = (df['ma9']  > df['ma20']).astype(int)
    df['ma20_above_ma50']  = (df['ma20'] > df['ma50']).astype(int)
    df['ma50_above_ma200'] = (df['ma50'] > df['ma200']).astype(int)

    # 기울기 (최근 5봉 변화율)
    df['ma20_slope'] = df['ma20'].pct_change(5) * 100
    df['ma50_slope'] = df['ma50'].pct_change(5) * 100

    df['above_ma20'] = (c > df['ma20']).astype(int)
    df['above_ma50'] = (c > df['ma50']).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 5. VWAP (분봉 핵심 지표)
# ─────────────────────────────────────────────────────────

def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP: 당일 기준으로 리셋되는 거래량 가중 평균가
    단기 트레이더가 가장 많이 보는 지표 중 하나
    """
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    # 날짜별 누적 VWAP 계산
    df['date'] = df.index.date
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol']    = df.groupby('date')['volume'].cumsum()
    df['vwap']       = df['cum_tp_vol'] / df['cum_vol'].replace(0, np.nan)

    df['dist_vwap']     = (df['close'] - df['vwap']) / df['vwap'] * 100  # VWAP 대비 이격도
    df['above_vwap']    = (df['close'] > df['vwap']).astype(int)
    df['vwap_cross_up'] = ((df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))).astype(int)
    df['vwap_cross_dn'] = ((df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))).astype(int)

    # 중간 계산 컬럼 제거
    df.drop(columns=['typical_price', 'tp_vol', 'cum_tp_vol', 'cum_vol', 'date'], inplace=True)

    return df


# ─────────────────────────────────────────────────────────
# 6. 장중 시간대 피처 (분봉 전용)
# ─────────────────────────────────────────────────────────

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    장중 시간대별 특성 반영
    - 오전 30분: 갭/급등 구간
    - 점심: 거래 감소 횡보
    - 마감 30분: 포지션 청산 급변동
    """
    hour   = np.array(df.index.hour)
    minute = np.array(df.index.minute)
    total_min = hour * 60 + minute  # 분 단위 절대시간

    # 장 시작 후 경과 분 (9:30 = 570분 기준)
    df['mins_from_open']  = np.clip(total_min - 570, 0, None)
    # 장 마감까지 남은 분 (16:00 = 960분 기준)
    df['mins_to_close']   = np.clip(960 - total_min, 0, None)

    # 시간대 구분 (원핫)
    df['session_open']    = ((total_min >= 570) & (total_min < 600)).astype(int)   # 9:30~10:00
    df['session_morning'] = ((total_min >= 600) & (total_min < 720)).astype(int)   # 10:00~12:00
    df['session_lunch']   = ((total_min >= 720) & (total_min < 810)).astype(int)   # 12:00~13:30
    df['session_afternoon']= ((total_min >= 810) & (total_min < 900)).astype(int)  # 13:30~15:00
    df['session_close']   = ((total_min >= 900) & (total_min < 960)).astype(int)   # 15:00~16:00

    # 요일 (월=0, 금=4)
    df['weekday'] = df.index.weekday
    df['is_monday'] = (df['weekday'] == 0).astype(int)
    df['is_friday'] = (df['weekday'] == 4).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 7. 모멘텀 지표
# ─────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']

    # RSI (14봉 = 약 70분)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi_overbought'] = (df['rsi14'] > 70).astype(int)
    df['rsi_oversold']   = (df['rsi14'] < 30).astype(int)

    # MACD (12, 26, 9봉)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']              = ema12 - ema26
    df['macd_signal']       = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']         = df['macd'] - df['macd_signal']
    df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_hist_rising']  = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    # Stochastic
    lowest14  = l.rolling(14).min()
    highest14 = h.rolling(14).max()
    df['stoch_k']          = (c - lowest14) / (highest14 - lowest14).replace(0, np.nan) * 100
    df['stoch_d']          = df['stoch_k'].rolling(3).mean()
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold']   = (df['stoch_k'] < 20).astype(int)

    # 단기 수익률
    for n in [1, 3, 6, 12]:  # 5분, 15분, 30분, 1시간
        df[f'roc_{n}'] = c.pct_change(n) * 100

    return df


# ─────────────────────────────────────────────────────────
# 8. 거래량 피처
# ─────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    v = df['volume']
    c = df['close']

    df['vol_ma12']  = v.rolling(12).mean()   # 1시간 평균
    df['vol_ma50']  = v.rolling(50).mean()   # 약 4시간 평균
    df['vol_ratio_1h']  = v / df['vol_ma12'].replace(0, np.nan)
    df['vol_ratio_4h']  = v / df['vol_ma50'].replace(0, np.nan)
    df['vol_spike']     = (df['vol_ratio_1h'] > 2.5).astype(int)

    # 당일 누적 거래량 대비 현재 거래량 위치
    df['date'] = df.index.date
    df['cum_vol_today'] = df.groupby('date')['volume'].cumsum()
    df.drop(columns=['date'], inplace=True)

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if c.iloc[i] > c.iloc[i-1]:
            obv.append(obv[-1] + v.iloc[i])
        elif c.iloc[i] < c.iloc[i-1]:
            obv.append(obv[-1] - v.iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_slope'] = pd.Series(obv, index=df.index).pct_change(6) * 100

    df['bull_vol'] = np.where(df['is_bullish'] == 1, df['vol_ratio_1h'], 0)
    df['bear_vol'] = np.where(df['is_bullish'] == 0, df['vol_ratio_1h'], 0)

    return df


# ─────────────────────────────────────────────────────────
# 9. 볼린저 밴드 & 변동성
# ─────────────────────────────────────────────────────────

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']

    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper']       = ma20 + 2 * std20
    df['bb_lower']       = ma20 - 2 * std20
    df['bb_width']       = (df['bb_upper'] - df['bb_lower']) / ma20
    df['bb_pct']         = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_above_upper'] = (c > df['bb_upper']).astype(int)
    df['bb_below_lower'] = (c < df['bb_lower']).astype(int)

    # ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df['atr14']      = tr.rolling(14).mean()
    df['atr_ratio']  = df['atr14'] / c * 100

    return df


# ─────────────────────────────────────────────────────────
# 10. 단기 지지/저항
# ─────────────────────────────────────────────────────────

def add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']

    # 분봉 기준: 12봉=1시간, 50봉=4시간, 78봉=하루(6.5h)
    for w in [12, 50, 78]:
        label = {12: '1h', 50: '4h', 78: '1d'}[w]
        df[f'high_{label}'] = h.rolling(w).max()
        df[f'low_{label}']  = l.rolling(w).min()
        rng = df[f'high_{label}'] - df[f'low_{label}']
        df[f'pos_in_range_{label}'] = (c - df[f'low_{label}']) / rng.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────
# 11. 래그 피처
# ─────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']

    # 과거 1~6봉 수익률 (5분 ~ 30분)
    for lag in range(1, 7):
        df[f'return_lag{lag}'] = c.pct_change(lag) * 100

    # RSI 래그
    for lag in [1, 3]:
        df[f'rsi_lag{lag}'] = df['rsi14'].shift(lag)

    # MACD 히스토그램 래그
    df['macd_hist_lag1'] = df['macd_hist'].shift(1)

    return df


# ─────────────────────────────────────────────────────────
# 12. 라벨 생성
# ─────────────────────────────────────────────────────────

def add_labels(df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """
    다음 5분봉 방향 라벨
    threshold: 0.1% (5분봉은 변동폭이 작아서 일봉보다 낮게 설정)
    """
    future_return = df['close'].shift(-1) / df['close'] - 1

    df['label'] = np.where(
        future_return >  threshold,  1,
        np.where(
        future_return < -threshold, -1,
                                     0)
    )
    df['label_name']       = df['label'].map({1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'})
    df['future_return_pct']= future_return * 100

    return df


# ─────────────────────────────────────────────────────────
# 13. 피처 컬럼 목록
# ─────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {
        'open', 'high', 'low', 'close', 'volume',
        'label', 'label_name', 'future_return_pct',
        # 중간 계산값
        'ma9', 'ma20', 'ma50', 'ma200',
        'bb_upper', 'bb_lower',
        'macd_signal', 'stoch_d',
        'vol_ma12', 'vol_ma50',
        'atr14', 'obv', 'vwap',
        'high_1h', 'low_1h', 'high_4h', 'low_4h', 'high_1d', 'low_1d',
    }
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


# ─────────────────────────────────────────────────────────
# 14. 전체 파이프라인
# ─────────────────────────────────────────────────────────

def run_pipeline(ticker: str, threshold: float = 0.001) -> pd.DataFrame:
    print(f"\n{'='*52}")
    print(f"  {ticker}  5분봉 파이프라인")
    print(f"{'='*52}")

    df = fetch_5m_ohlcv(ticker)

    print(f"  피처 생성 중...")
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
    df = add_labels(df, threshold=threshold)

    before = len(df)
    df.dropna(inplace=True)
    print(f"  NaN 제거: {before} → {len(df)}행")

    dist  = df['label_name'].value_counts()
    total = len(df)
    print(f"\n  라벨 분포:")
    for name, cnt in dist.items():
        bar = '█' * int(cnt / total * 30)
        print(f"    {name:8s}: {cnt:5d}행 ({cnt/total*100:.1f}%) {bar}")

    feature_cols = get_feature_columns(df)
    print(f"\n  생성된 피처: {len(feature_cols)}개")

    return df


def save_features(df: pd.DataFrame, ticker: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{ticker}_5m_features.csv")
    df.to_csv(path)

    feature_cols = get_feature_columns(df)
    df[feature_cols].to_csv(os.path.join(output_dir, f"{ticker}_5m_X.csv"))
    df[['label', 'label_name', 'future_return_pct']].to_csv(os.path.join(output_dir, f"{ticker}_5m_y.csv"))

    print(f"\n  저장: {path}")
    return path


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='5분봉 ML 데이터 파이프라인')
    parser.add_argument('--ticker',    nargs='+', default=['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META'], help='종목 티커')
    parser.add_argument('--threshold', type=float, default=0.001, help='라벨 임계값 (기본 0.1%%)')
    parser.add_argument('--output',    default='output', help='출력 디렉토리')
    args = parser.parse_args()

    print(f"\n5분봉 ML 데이터 파이프라인")
    print(f"종목: {args.ticker}")
    print(f"라벨 임계값: {args.threshold*100:.2f}%")

    all_dfs = []
    for ticker in args.ticker:
        try:
            df = run_pipeline(ticker, threshold=args.threshold)
            save_features(df, ticker, output_dir=args.output)
            df['ticker'] = ticker
            all_dfs.append(df)
        except Exception as e:
            print(f"  [{ticker}] 오류: {e}")

    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs)
        path = os.path.join(args.output, 'combined_5m_features.csv')
        combined.to_csv(path)
        total = len(combined)
        print(f"\n{'='*52}")
        print(f"  합산 완료: {total:,}행 ({len(all_dfs)}개 종목)")
        print(f"  저장: {path}")
        print(f"  → 다음 단계: python train_model.py")
        print(f"{'='*52}\n")


if __name__ == '__main__':
    main()
