"""
5분봉 차트 ML 파이프라인 v3 - 데이터 수집 & 피처 생성
=====================================================
변경사항 (v2 → v3):
  - 폴더 경로 chart_ml_v2 → chart_ml_v4
  - CNN 학습용 시퀀스 데이터 추가 저장 (output/combined_5m_sequences.npy)
  - 나머지 로직 v2와 동일

사용법:
  pip install yfinance pandas numpy scikit-learn --break-system-packages

  # 기본 30종목 실행
  python data_pipeline_5m.py

  # 종목 직접 지정
  python data_pipeline_5m.py --ticker AAPL MSFT NVDA

출력:
  output/AAPL_5m_features.csv
  output/combined_5m_features.csv  (여러 종목 합본 → train_model.py 입력)

Yahoo Finance 5분봉 제한:
  - 최대 60일치만 제공
  - 60일 × 6.5시간 × 12봉 = 약 4,680개/종목
  - 30개 종목 합산 시 약 195,000개 → LSTM 실험 가능
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────
# 1. 데이터 수집
# ─────────────────────────────────────────────────────────

def fetch_premarket_features(ticker: str) -> pd.DataFrame:
    """
    프리마켓 데이터 수집 후 날짜별 피처로 변환
    Yahoo Finance 5분봉에서 04:00~09:29 구간 추출

    반환: 날짜(date)를 인덱스로 하는 DataFrame
      - premarket_gap       : 전일 종가 대비 프리마켓 시작가 갭 (%)
      - premarket_return    : 프리마켓 수익률 (시작→마지막, %)
      - premarket_volume    : 프리마켓 총 거래량
      - premarket_high      : 프리마켓 고가
      - premarket_low       : 프리마켓 저가
      - premarket_range     : 프리마켓 변동폭 (고가-저가, %)
      - premarket_vs_open   : 프리마켓 마지막가 vs 본장 시가 차이 (%)
    """
    print(f"  [{ticker}] 프리마켓 데이터 수집 중...")
    try:
        df = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')

        # 프리마켓 구간만 추출 (04:00 ~ 09:29)
        pm = df.iloc[df.index.indexer_between_time('04:00', '09:29')].copy()
        if pm.empty:
            return pd.DataFrame()

        pm['date'] = pm.index.date

        # 본장 시가 (당일 09:30 첫 봉)
        mkt = df.iloc[df.index.indexer_between_time('09:30', '09:30')].copy()
        mkt['date'] = mkt.index.date
        mkt_open = mkt.groupby('date')['open'].first()

        # 전일 종가 (본장 마지막 봉)
        reg = df.iloc[df.index.indexer_between_time('09:30', '15:55')].copy()
        reg['date'] = reg.index.date
        prev_close = reg.groupby('date')['close'].last().shift(1)

        # 날짜별 프리마켓 집계
        result = pd.DataFrame(index=pm['date'].unique())
        result.index = pd.to_datetime(result.index)

        pm_first  = pm.groupby('date')['open'].first()
        pm_last   = pm.groupby('date')['close'].last()
        pm_high   = pm.groupby('date')['high'].max()
        pm_low    = pm.groupby('date')['low'].min()
        pm_volume = pm.groupby('date')['volume'].sum()

        result.index = pd.to_datetime(result.index)
        prev_close.index = pd.to_datetime(prev_close.index)
        mkt_open.index   = pd.to_datetime(mkt_open.index)

        result['premarket_gap']      = (pm_first / prev_close - 1) * 100
        result['premarket_return']   = (pm_last  / pm_first   - 1) * 100
        result['premarket_volume']   = pm_volume
        result['premarket_high']     = pm_high
        result['premarket_low']      = pm_low
        result['premarket_range']    = (pm_high - pm_low) / pm_low * 100
        result['premarket_vs_open']  = (pm_last / mkt_open - 1) * 100

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)

        print(f"  [{ticker}] 프리마켓 피처 생성 완료 ({len(result)}일)")
        return result

    except Exception as e:
        print(f"  [{ticker}] 프리마켓 수집 실패 (기본값 0으로 대체): {e}")
        return pd.DataFrame()


def fetch_aftermarket_features(ticker: str) -> pd.DataFrame:
    """
    애프터마켓 데이터 수집 후 날짜별 피처로 변환
    Yahoo Finance 5분봉에서 16:00~20:00 구간 추출
    → 다음날 본장에 병합 (어제 애프터마켓이 오늘 본장에 영향)

    반환: 날짜(date)를 인덱스로 하는 DataFrame (다음날 기준)
      - aftermarket_return   : 애프터마켓 수익률 (본장 종가→애프터 마지막, %)
      - aftermarket_volume   : 애프터마켓 총 거래량
      - aftermarket_range    : 애프터마켓 변동폭 (고가-저가, %)
      - aftermarket_vs_close : 애프터마켓 마지막가 vs 본장 종가 차이 (%)
      - has_earnings_move    : 애프터마켓 변동폭 > 3% (어닝/이벤트 감지, 0/1)
    """
    print(f"  [{ticker}] 애프터마켓 데이터 수집 중...")
    try:
        df = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')

        # 애프터마켓 구간 추출 (16:00 ~ 19:55)
        am = df.iloc[df.index.indexer_between_time('16:00', '19:55')].copy()
        if am.empty:
            return pd.DataFrame()

        am['date'] = am.index.date

        # 본장 종가 (당일 15:55 마지막 봉)
        reg = df.iloc[df.index.indexer_between_time('09:30', '15:55')].copy()
        reg['date'] = reg.index.date
        mkt_close = reg.groupby('date')['close'].last()

        # 날짜별 애프터마켓 집계
        result = pd.DataFrame(index=am['date'].unique())
        result.index = pd.to_datetime(result.index)

        am_first  = am.groupby('date')['open'].first()
        am_last   = am.groupby('date')['close'].last()
        am_high   = am.groupby('date')['high'].max()
        am_low    = am.groupby('date')['low'].min()
        am_volume = am.groupby('date')['volume'].sum()

        mkt_close.index = pd.to_datetime(mkt_close.index)

        result['aftermarket_return']   = (am_last / mkt_close - 1) * 100
        result['aftermarket_volume']   = am_volume
        result['aftermarket_range']    = (am_high - am_low) / am_low * 100
        result['aftermarket_vs_close'] = (am_last / mkt_close - 1) * 100
        result['has_earnings_move']    = (result['aftermarket_range'] > 3.0).astype(int)

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)

        # 다음날 본장에 영향 → 인덱스를 하루 뒤로 shift
        result.index = result.index + pd.Timedelta(days=1)
        # 주말 건너뛰기 (금→월)
        result.index = result.index.map(
            lambda d: d + pd.Timedelta(days=2) if d.weekday() == 5  # 토 → 월
            else d + pd.Timedelta(days=1) if d.weekday() == 6       # 일 → 월
            else d
        )

        print(f"  [{ticker}] 애프터마켓 피처 생성 완료 ({len(result)}일)")
        return result

    except Exception as e:
        print(f"  [{ticker}] 애프터마켓 수집 실패 (기본값 0으로 대체): {e}")
        return pd.DataFrame()


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
    df['upper_wick']       = h - np.maximum(o, c)
    df['lower_wick']       = np.minimum(o, c) - l

    # 비율 계산용 임시 변수 (피처로 저장 안 함)
    _rng = (h - l).replace(0, np.nan)
    df['body_ratio']       = df['body'].abs() / _rng
    df['upper_wick_ratio'] = df['upper_wick'] / _rng
    df['lower_wick_ratio'] = df['lower_wick'] / _rng
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

    # engulf 패턴만 유지 (5분봉에서 신뢰도 있는 패턴)
    df['pat_bullish_engulf'] = ((df['is_bullish'] == 1) & (df['is_bullish'].shift(1) == 0) & (o < c.shift(1)) & (c > o.shift(1))).astype(int)
    df['pat_bearish_engulf'] = ((df['is_bullish'] == 0) & (df['is_bullish'].shift(1) == 1) & (o > c.shift(1)) & (c < o.shift(1))).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 4. 이동평균 (분봉 적합한 단위)
# ─────────────────────────────────────────────────────────

def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']

    for w in [9, 20, 50, 200]:
        df[f'ma{w}']      = c.rolling(w).mean()
        df[f'dist_ma{w}'] = (c - df[f'ma{w}']) / df[f'ma{w}'] * 100

    df['ma9_above_ma20']  = (df['ma9']  > df['ma20']).astype(int)
    df['ma20_above_ma50'] = (df['ma20'] > df['ma50']).astype(int)

    # 기울기 (최근 5봉 변화율)
    df['ma20_slope'] = df['ma20'].pct_change(5) * 100
    df['ma50_slope'] = df['ma50'].pct_change(5) * 100

    df['above_ma20'] = (c > df['ma20']).astype(int)

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
    hour   = np.array(df.index.hour)
    minute = np.array(df.index.minute)
    total_min = hour * 60 + minute

    df['mins_from_open']   = np.clip(total_min - 570, 0, None)
    df['mins_to_close']    = np.clip(960 - total_min, 0, None)

    df['session_open']      = ((total_min >= 570) & (total_min < 600)).astype(int)
    df['session_morning']   = ((total_min >= 600) & (total_min < 720)).astype(int)
    df['session_afternoon'] = ((total_min >= 810) & (total_min < 900)).astype(int)
    df['session_close']     = ((total_min >= 900) & (total_min < 960)).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 7. 모멘텀 지표
# ─────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close']
    h = df['high']
    l = df['low']

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # MACD (hist와 파생 피처만 유지)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    _macd               = ema12 - ema26
    df['macd_signal']   = _macd.ewm(span=9, adjust=False).mean()
    df['macd_hist']     = _macd - df['macd_signal']
    df['macd_above_signal'] = (_macd > df['macd_signal']).astype(int)
    df['macd_hist_rising']  = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    # Stochastic
    lowest14  = l.rolling(14).min()
    highest14 = h.rolling(14).max()
    df['stoch_k']          = (c - lowest14) / (highest14 - lowest14).replace(0, np.nan) * 100
    df['stoch_d']          = df['stoch_k'].rolling(3).mean()
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold']   = (df['stoch_k'] < 20).astype(int)

    # 단기 수익률
    for n in [1, 3, 6, 12]:
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
# 12. Put/Call Ratio 피처
# ─────────────────────────────────────────────────────────

def fetch_vix_features() -> pd.DataFrame:
    """
    VIX (변동성 지수) 수집 — ^VIX (yfinance에서 안정적으로 지원)

    반환: 날짜 인덱스 DataFrame
      - vix_close    : VIX 종가 (시장 공포 수준)
      - vix_change   : 전일 대비 변화율 (%)
      - vix_signal   : 1=고공포(VIX>25), -1=저공포(VIX<15), 0=중립
    """
    print(f"  [VIX] 수집 중...")
    try:
        raw = yf.download('^VIX', period='90d', interval='1d',
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("VIX 데이터 없음")

        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]

        result = pd.DataFrame(index=raw.index)
        result['vix_close']  = raw['close'].values
        result['vix_change'] = raw['close'].pct_change() * 100
        result['vix_signal'] = np.where(
            result['vix_close'] > 25,  1,   # 고공포 → 반등 가능성
            np.where(
            result['vix_close'] < 15, -1,   # 저공포 → 과열 주의
                                       0)
        )

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.ffill(inplace=True)
        result['vix_close']  = result['vix_close'].fillna(20.0)
        result['vix_change'] = result['vix_change'].fillna(0.0)
        result['vix_signal'] = result['vix_signal'].fillna(0.0)

        print(f"  [VIX] 수집 완료 ({len(result)}일, 최신 VIX={result['vix_close'].iloc[-1]:.2f})")
        return result

    except Exception as e:
        print(f"  [VIX] 수집 실패 (기본값으로 대체): {e}")
        return pd.DataFrame()


def add_vix_features(df: pd.DataFrame, vix_features: pd.DataFrame) -> pd.DataFrame:
    """날짜별 VIX 피처를 본장 5분봉 전체에 병합"""
    vix_cols = ['vix_close', 'vix_change', 'vix_signal']

    if vix_features.empty:
        df['vix_close']  = 20.0
        df['vix_change'] = 0.0
        df['vix_signal'] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    vix_features.index = pd.to_datetime(vix_features.index)

    if vix_features.index.tz is not None:
        vix_features.index = vix_features.index.tz_localize(None)

    df = df.merge(
        vix_features[vix_cols],
        left_on='_date', right_index=True, how='left'
    )
    df.drop(columns=['_date'], inplace=True)
    df['vix_close']  = df['vix_close'].ffill().fillna(20.0)
    df['vix_change'] = df['vix_change'].ffill().fillna(0.0)
    df['vix_signal'] = df['vix_signal'].ffill().fillna(0.0)
    return df


def fetch_gold_features() -> pd.DataFrame:
    """
    금 선물 (GC=F) 일별 데이터 수집
      - gold_close  : 금 종가
      - gold_change : 전일 대비 변화율 (%)
      - gold_signal : 1=급등(+1%↑), -1=급락(-1%↓), 0=중립
    금 급등 = 안전자산 선호 = 위험자산 회피 경향
    """
    print(f"  [GOLD] 수집 중...")
    try:
        raw = yf.download('GC=F', period='90d', interval='1d',
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("금 데이터 없음")

        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]

        result = pd.DataFrame(index=raw.index)
        result['gold_close']  = raw['close'].values
        result['gold_change'] = raw['close'].pct_change() * 100
        result['gold_signal'] = np.where(
            result['gold_change'] >  1.0,  1,
            np.where(
            result['gold_change'] < -1.0, -1, 0)
        )

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.ffill(inplace=True)
        result['gold_close']  = result['gold_close'].fillna(result['gold_close'].median())
        result['gold_change'] = result['gold_change'].fillna(0.0)
        result['gold_signal'] = result['gold_signal'].fillna(0.0)

        print(f"  [GOLD] 수집 완료 ({len(result)}일, 최신 ${result['gold_close'].iloc[-1]:.1f})")
        return result

    except Exception as e:
        print(f"  [GOLD] 수집 실패 (기본값으로 대체): {e}")
        return pd.DataFrame()


def add_gold_features(df: pd.DataFrame, gold_features: pd.DataFrame) -> pd.DataFrame:
    """날짜별 금 피처를 본장 5분봉 전체에 병합"""
    if gold_features.empty:
        df['gold_close']  = 2000.0
        df['gold_change'] = 0.0
        df['gold_signal'] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    gold_features.index = pd.to_datetime(gold_features.index)
    if gold_features.index.tz is not None:
        gold_features.index = gold_features.index.tz_localize(None)

    df = df.merge(
        gold_features[['gold_close', 'gold_change', 'gold_signal']],
        left_on='_date', right_index=True, how='left'
    )
    df.drop(columns=['_date'], inplace=True)
    df['gold_close']  = df['gold_close'].ffill().fillna(2000.0)
    df['gold_change'] = df['gold_change'].ffill().fillna(0.0)
    df['gold_signal'] = df['gold_signal'].ffill().fillna(0.0)
    return df


def fetch_oil_features() -> pd.DataFrame:
    """
    WTI 원유 선물 (CL=F) 일별 데이터 수집
      - oil_close  : WTI 종가
      - oil_change : 전일 대비 변화율 (%)
      - oil_signal : 1=급등(+2%↑), -1=급락(-2%↓), 0=중립
    유가 급등 = 인플레 우려 / 에너지 섹터 강세
    유가 급락 = 경기침체 우려 / 위험자산 약세
    """
    print(f"  [OIL] 수집 중...")
    try:
        raw = yf.download('CL=F', period='90d', interval='1d',
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("유가 데이터 없음")

        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]

        result = pd.DataFrame(index=raw.index)
        result['oil_close']  = raw['close'].values
        result['oil_change'] = raw['close'].pct_change() * 100
        result['oil_signal'] = np.where(
            result['oil_change'] >  2.0,  1,
            np.where(
            result['oil_change'] < -2.0, -1, 0)
        )

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.ffill(inplace=True)
        result['oil_close']  = result['oil_close'].fillna(result['oil_close'].median())
        result['oil_change'] = result['oil_change'].fillna(0.0)
        result['oil_signal'] = result['oil_signal'].fillna(0.0)

        print(f"  [OIL] 수집 완료 ({len(result)}일, 최신 ${result['oil_close'].iloc[-1]:.1f})")
        return result

    except Exception as e:
        print(f"  [OIL] 수집 실패 (기본값으로 대체): {e}")
        return pd.DataFrame()


def add_oil_features(df: pd.DataFrame, oil_features: pd.DataFrame) -> pd.DataFrame:
    """날짜별 유가 피처를 본장 5분봉 전체에 병합"""
    if oil_features.empty:
        df['oil_close']  = 75.0
        df['oil_change'] = 0.0
        df['oil_signal'] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    oil_features.index = pd.to_datetime(oil_features.index)
    if oil_features.index.tz is not None:
        oil_features.index = oil_features.index.tz_localize(None)

    df = df.merge(
        oil_features[['oil_close', 'oil_change', 'oil_signal']],
        left_on='_date', right_index=True, how='left'
    )
    df.drop(columns=['_date'], inplace=True)
    df['oil_close']  = df['oil_close'].ffill().fillna(75.0)
    df['oil_change'] = df['oil_change'].ffill().fillna(0.0)
    df['oil_signal'] = df['oil_signal'].ffill().fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────
# 13. DXY (달러 인덱스)
# ─────────────────────────────────────────────────────────

def fetch_dxy_features() -> pd.DataFrame:
    """
    달러 인덱스 (DX=F) 일별 데이터 수집
      - dxy_close  : DXY 종가
      - dxy_change : 전일 대비 변화율 (%)
      - dxy_signal : 1=달러강세(+0.5%↑), -1=달러약세(-0.5%↓), 0=중립

    달러 강세 → 기술주/성장주 약세, 원자재 약세
    달러 약세 → 위험자산 선호, 신흥국/원자재 강세
    """
    print(f"  [DXY] 수집 중...")
    try:
        raw = yf.download('DX=F', period='90d', interval='1d',
                          progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("DXY 데이터 없음")

        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]

        result = pd.DataFrame(index=raw.index)
        result['dxy_close']  = raw['close'].values
        result['dxy_change'] = raw['close'].pct_change() * 100
        result['dxy_signal'] = np.where(
            result['dxy_change'] >  0.5,  1,   # 달러 강세
            np.where(
            result['dxy_change'] < -0.5, -1,   # 달러 약세
                                          0)
        )

        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.ffill(inplace=True)
        result['dxy_close']  = result['dxy_close'].fillna(result['dxy_close'].median())
        result['dxy_change'] = result['dxy_change'].fillna(0.0)
        result['dxy_signal'] = result['dxy_signal'].fillna(0.0)

        print(f"  [DXY] 수집 완료 ({len(result)}일, 최신 DXY={result['dxy_close'].iloc[-1]:.2f})")
        return result

    except Exception as e:
        print(f"  [DXY] 수집 실패 (기본값으로 대체): {e}")
        return pd.DataFrame()


def add_dxy_features(df: pd.DataFrame, dxy_features: pd.DataFrame) -> pd.DataFrame:
    """날짜별 DXY 피처를 본장 5분봉 전체에 병합"""
    if dxy_features.empty:
        df['dxy_close']  = 104.0
        df['dxy_change'] = 0.0
        df['dxy_signal'] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    dxy_features.index = pd.to_datetime(dxy_features.index)
    if dxy_features.index.tz is not None:
        dxy_features.index = dxy_features.index.tz_localize(None)

    df = df.merge(
        dxy_features[['dxy_close', 'dxy_change', 'dxy_signal']],
        left_on='_date', right_index=True, how='left'
    )
    df.drop(columns=['_date'], inplace=True)
    df['dxy_close']  = df['dxy_close'].ffill().fillna(104.0)
    df['dxy_change'] = df['dxy_change'].ffill().fillna(0.0)
    df['dxy_signal'] = df['dxy_signal'].ffill().fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────
# 14. 섹터 더미 변수
# ─────────────────────────────────────────────────────────

# 종목 → 섹터 매핑
SECTOR_MAP = {
    # 테크
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech',
    'META': 'tech', 'AMZN': 'tech', 'NFLX': 'tech',
    'ORCL': 'tech', 'CRM':  'tech', 'UBER': 'tech',
    # 반도체
    'NVDA': 'semicon', 'AMD':  'semicon', 'INTC': 'semicon',
    'QCOM': 'semicon', 'AVGO': 'semicon',
    'MU':   'semicon', 'TSM':  'semicon',
    # 고변동성
    'TSLA': 'volatile', 'COIN': 'volatile', 'PLTR': 'volatile',
    'IREN': 'volatile', 'APP':  'volatile', 'IONQ': 'volatile',
    # 금융
    'JPM':  'finance', 'BAC':  'finance', 'GS':   'finance', 'V': 'finance',
    'MS':   'finance', 'SCHW': 'finance', 'BLK':  'finance',
    # 헬스케어
    'JNJ':  'health', 'UNH': 'health', 'LLY':  'health',
    'MRNA': 'health', 'ABBV': 'health',
    # 에너지
    'XOM': 'energy', 'CVX': 'energy',
    'OXY': 'energy', 'SLB': 'energy',
    # ETF
    'SPY': 'etf', 'QQQ': 'etf', 'IWM': 'etf',
    # 소비재
    'WMT':  'consumer', 'COST': 'consumer', 'NKE': 'consumer',
    'SHOP': 'consumer', 'KO':   'consumer',
    # 산업재 (신규)
    'CAT': 'industrial',
}

SECTORS = ['tech', 'semicon', 'volatile', 'finance',
           'health', 'energy', 'etf', 'consumer', 'industrial']


def add_sector_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    종목에 해당하는 섹터 더미 변수 추가
    sector_tech, sector_semicon, ... 각 0 또는 1
    """
    sector = SECTOR_MAP.get(ticker.upper(), 'tech')  # 미등록 종목은 tech로 기본값

    for s in SECTORS:
        df[f'sector_{s}'] = 1 if s == sector else 0

    return df


# ─────────────────────────────────────────────────────────
# 15. 상호작용 피처 (거시지표 × 섹터)
# ─────────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    거시지표 × 섹터 상호작용 피처
    XGBoost 트리가 섹터별 거시지표 반응을 명시적으로 학습할 수 있도록 함
    """
    # 기존 4개
    df['oil_x_energy']   = df['oil_change']  * df['sector_energy']    # 유가 × 에너지
    df['oil_x_semicon']  = df['oil_change']  * df['sector_semicon']   # 유가 × 반도체 (음의 상관)
    df['dxy_x_tech']     = df['dxy_change']  * df['sector_tech']      # 달러 × 기술주
    df['vix_x_volatile'] = df['vix_close']   * df['sector_volatile']  # VIX × 고변동성

    # 신규 8개
    df['vix_x_finance']  = df['vix_change']  * df['sector_finance']   # 공포 시 금융주 영향
    df['vix_x_health']   = df['vix_change']  * df['sector_health']    # 공포 시 헬스케어 방어
    df['gold_x_volatile']= df['gold_change'] * df['sector_volatile']  # 금 급등 시 고변동성 반응
    df['gold_x_finance'] = df['gold_change'] * df['sector_finance']   # 금 급등 시 금융주 영향
    df['dxy_x_energy']   = df['dxy_change']  * df['sector_energy']    # 달러 강세 시 에너지 영향
    df['dxy_x_semicon']  = df['dxy_change']  * df['sector_semicon']   # 달러 강세 시 반도체 영향
    df['oil_x_consumer'] = df['oil_change']  * df['sector_consumer']  # 유가 시 소비재 영향
    df['vix_x_etf']      = df['vix_change']  * df['sector_etf']       # 공포 시 ETF 흐름

    return df

def add_premarket_features(df: pd.DataFrame, pm_features: pd.DataFrame) -> pd.DataFrame:
    """
    날짜별 프리마켓 피처를 본장 5분봉 전체에 병합
    당일 모든 봉에 동일한 프리마켓 값을 붙임
    프리마켓 데이터 없으면 0으로 채움
    """
    pm_cols = [
        'premarket_gap', 'premarket_return', 'premarket_volume',
        'premarket_high', 'premarket_low', 'premarket_range', 'premarket_vs_open'
    ]

    if pm_features.empty:
        for col in pm_cols:
            df[col] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    pm_features.index = pd.to_datetime(pm_features.index)

    df = df.merge(
        pm_features[pm_cols],
        left_on='_date',
        right_index=True,
        how='left'
    )
    df.drop(columns=['_date'], inplace=True)

    for col in pm_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[pm_cols] = df[pm_cols].fillna(0)

    return df


def add_aftermarket_features(df: pd.DataFrame, am_features: pd.DataFrame) -> pd.DataFrame:
    """
    날짜별 애프터마켓 피처를 다음날 본장 5분봉 전체에 병합
    애프터마켓 데이터 없으면 0으로 채움
    """
    am_cols = [
        'aftermarket_return', 'aftermarket_volume',
        'aftermarket_range', 'aftermarket_vs_close', 'has_earnings_move'
    ]

    if am_features.empty:
        for col in am_cols:
            df[col] = 0.0
        return df

    df = df.copy()
    df['_date'] = pd.to_datetime(df.index.date)
    am_features.index = pd.to_datetime(am_features.index)

    df = df.merge(
        am_features[am_cols],
        left_on='_date',
        right_index=True,
        how='left'
    )
    df.drop(columns=['_date'], inplace=True)

    for col in am_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[am_cols] = df[am_cols].fillna(0)

    return df
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
        # 거시지표 절대값 (변화율/신호로 충분)
        'gold_close', 'oil_close', 'dxy_close',
        # vix_close는 vix_x_volatile 상호작용 피처에 사용하므로 유지
    }
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


# ─────────────────────────────────────────────────────────
# 14. 전체 파이프라인
# ─────────────────────────────────────────────────────────

def run_pipeline(ticker: str, threshold: float = 0.001,
                 vix_features:  pd.DataFrame = None,
                 gold_features: pd.DataFrame = None,
                 oil_features:  pd.DataFrame = None,
                 dxy_features:  pd.DataFrame = None) -> pd.DataFrame:
    print(f"\n{'='*52}")
    print(f"  {ticker}  5분봉 파이프라인")
    print(f"{'='*52}")

    df = fetch_5m_ohlcv(ticker)

    if vix_features  is None: vix_features  = pd.DataFrame()
    if gold_features is None: gold_features = pd.DataFrame()
    if oil_features  is None: oil_features  = pd.DataFrame()
    if dxy_features  is None: dxy_features  = pd.DataFrame()

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
    df = add_vix_features(df,  vix_features)    # VIX
    df = add_gold_features(df, gold_features)   # 금
    df = add_oil_features(df,  oil_features)    # 유가
    df = add_dxy_features(df,  dxy_features)    # 달러
    df = add_sector_features(df, ticker)        # 섹터 더미
    df = add_interaction_features(df)           # 상호작용 피처
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


def save_sequences(df: pd.DataFrame, output_dir: str = "output",
                   seq_len: int = 20):
    """
    CNN 학습용 시퀀스 데이터 생성 및 저장
    각 봉을 기준으로 이전 seq_len개 봉의 피처를 묶어 3D 배열로 만듦
    shape: (샘플수, seq_len, 피처수)

    combined_5m_features.csv 전체 기준으로 한 번만 호출
    """
    import numpy as np

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df['label'].copy()
    future_ret = df['future_return_pct'].copy()

    # 숫자 컬럼만 선택 (문자열 컬럼 제거)
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)

    X_arr = X.values
    y_arr = y.values
    r_arr = future_ret.values

    sequences, labels, returns = [], [], []
    for i in range(seq_len, len(X_arr)):
        sequences.append(X_arr[i - seq_len:i])  # (seq_len, n_features)
        labels.append(y_arr[i])
        returns.append(r_arr[i])

    seq_arr = np.array(sequences, dtype=np.float32)   # (N, seq_len, n_features)
    lbl_arr = np.array(labels,    dtype=np.int8)
    ret_arr = np.array(returns,   dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'sequences_X.npy'), seq_arr)
    np.save(os.path.join(output_dir, 'sequences_y.npy'), lbl_arr)
    np.save(os.path.join(output_dir, 'sequences_ret.npy'), ret_arr)

    print(f"\n  CNN 시퀀스 저장 완료")
    print(f"    sequences_X.npy  : {seq_arr.shape}  (샘플, {seq_len}봉, 피처)")
    print(f"    sequences_y.npy  : {lbl_arr.shape}")
    print(f"    sequences_ret.npy: {ret_arr.shape}")


# v4 기본 50종목
TICKERS_V3 = [
    # 테크 대형주 (6)
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX',
    # 테크 추가 (3)
    'ORCL', 'CRM', 'UBER',
    # 반도체 (5)
    'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO',
    # 반도체 추가 (2)
    'MU', 'TSM',
    # 고변동성 (3) ← MSTR 제거
    'TSLA', 'COIN', 'PLTR',
    # 고변동성 추가 (3)
    'IREN', 'APP', 'IONQ',
    # 금융 (4)
    'JPM', 'BAC', 'GS', 'V',
    # 금융 추가 (3)
    'MS', 'SCHW', 'BLK',
    # 헬스케어 (3)
    'JNJ', 'UNH', 'LLY',
    # 헬스케어 추가 (2)
    'MRNA', 'ABBV',
    # 에너지 (2)
    'XOM', 'CVX',
    # 에너지 추가 (2)
    'OXY', 'SLB',
    # ETF (3)
    'SPY', 'QQQ', 'IWM',
    # 소비재 (3)
    'WMT', 'COST', 'NKE',
    # 소비재 추가 (2)
    'SHOP', 'KO',
    # 산업재 신규 (1)
    'CAT',
]


def main():
    parser = argparse.ArgumentParser(description='5분봉 ML 데이터 파이프라인 v3')
    parser.add_argument('--ticker',    nargs='+', default=TICKERS_V3, help='종목 티커 (기본: 30종목)')
    parser.add_argument('--threshold', type=float, default=0.0015, help='라벨 임계값 (기본 0.15%%)')
    parser.add_argument('--output',    default='output', help='출력 디렉토리')
    parser.add_argument('--delay',     type=float, default=2.0, help='종목 간 API 호출 딜레이(초), 기본 2.0')
    parser.add_argument('--seq-len',   type=int, default=20, help='CNN 시퀀스 길이 (기본 20봉)')
    args = parser.parse_args()

    print(f"\n5분봉 ML 데이터 파이프라인 v3")
    print(f"종목 수: {len(args.ticker)}개")
    print(f"종목: {args.ticker}")
    print(f"라벨 임계값: {args.threshold*100:.3f}%")
    print(f"API 딜레이: {args.delay}초")
    print(f"CNN 시퀀스 길이: {args.seq_len}봉")

    # 거시경제 지표 — 시장 공통, 전 종목 1회만 수집
    vix_features  = fetch_vix_features()
    gold_features = fetch_gold_features()
    oil_features  = fetch_oil_features()
    dxy_features  = fetch_dxy_features()

    all_dfs = []
    for i, ticker in enumerate(args.ticker):
        try:
            df = run_pipeline(ticker, threshold=args.threshold,
                              vix_features=vix_features,
                              gold_features=gold_features,
                              oil_features=oil_features,
                              dxy_features=dxy_features)
            save_features(df, ticker, output_dir=args.output)
            df['ticker'] = ticker
            all_dfs.append(df)
            if i < len(args.ticker) - 1:
                print(f"  [{ticker}] 완료 → {args.delay}초 대기 후 다음 종목...")
                time.sleep(args.delay)
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
        print(f"  → 다음 단계: python train_model.py --data {path}")
        print(f"{'='*52}\n")

        # CNN 학습용 시퀀스 데이터 저장
        print(f"  CNN 시퀀스 생성 중 (seq_len={args.seq_len})...")
        save_sequences(combined, output_dir=args.output, seq_len=args.seq_len)


if __name__ == '__main__':
    main()
