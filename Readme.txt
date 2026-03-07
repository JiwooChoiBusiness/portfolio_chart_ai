Readme

바탕화면에 파일을 놓으세요.

1번(티커로 데이터 수집): data_pipeline_5m.py
cd C:\Users\pc\Desktop\chart_ml
py -3.12 c:/Users/pc/Desktop/chart_ml/data_pipeline_5m.py --ticker AAPL MSFT GOOGL AMZN META NVDA TSLA AMD NFLX JPM BAC GS SPY QQQ IWM XLK SOXX 
원하는 티커를 적으세요 


2번(모델 트레인) train_model.py
py -3.12 c:/Users/pc/Desktop/chart_ml/train_model.py

3번(예측 - 모델 트레인까지 완료했으면 티커만 바꿔서 실행하면 됨) predict.py
py -3.12 c:/Users/pc/Desktop/chart_ml/predict.py --ticker snxx

시각화
4번(모델 결과 시각화)visualize_model.py

cd C:\Users\pc\Desktop\chart_ml
py -3.12 visualize_model.py

5번(예측 결과 시각화)visualize_predict.py 3번 티커와 동일하게 하세요

cd C:\Users\pc\Desktop\chart_ml

py -3.12 visualize_predict.py --ticker soxl

-------------------
project3.bitcoin_dashboard 와는 별개입니다.
