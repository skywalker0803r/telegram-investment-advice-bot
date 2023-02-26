import yfinance as yf
from finta import TA
from keras.models import load_model
import pickle
import datetime as dt
import requests
import schedule
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#https://sharegpt.com/c/kZOYTX1

def send_to_telegram(message):
    apiToken = '5850662274:AAGeKZqM1JfQfh3CrSKG6BZ9pEvDajdBUqs'
    chatID = '1567262377'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

def process(df,min_max_scaler,model):
    df['RSI'] = TA.RSI(df)#1
    df['Williams %R'] = TA.WILLIAMS(df)#2
    df['SMA'] = TA.SMA(df)#3
    df['EMA'] = TA.EMA(df)#4
    df['WMA'] = TA.WMA(df)#5
    df['HMA'] = TA.HMA(df)#6
    df['TEMA'] = TA.TEMA(df)#7
    df['CCI'] = TA.CCI(df)#8
    df['CMO'] = TA.CMO(df)#9
    df['MACD'] = TA.MACD(df)['MACD'] - TA.MACD(df)['SIGNAL']#10
    df['PPO'] = TA.PPO(df)['PPO'] - TA.PPO(df)['SIGNAL']#11
    df['ROC'] = TA.ROC(df)#12
    df['CFI'] = TA.CFI(df)#13
    df['DMI'] = TA.DMI(df)['DI+'] - TA.DMI(df)['DI-']#14
    df['SAR'] = TA.SAR(df)#15
    df = df.dropna(axis=0)
    features = df.columns[-15:].tolist()
    Close = df[['Close']]
    df = df[features]
    # 數值轉換
    df[features] = min_max_scaler.transform(df[features])
    # 製作X
    days = 15
    start_index = 0
    end_index = len(df)-days
    Xs = []
    indexs = []
    for i in tqdm(range(start_index ,end_index+1 ,1)):
        X = df.iloc[i:i+days,:][features]
        X = np.array(X)
        Xs.append(X)
        indexs.append((df.iloc[[i]].index,df.iloc[[i+days-1]].index))
    Xs = np.array(Xs)
    # 模型預測
    answer = model.predict(Xs)
    answer = [ np.argmax(i) for i in answer]
    
    # 繪圖
    Close = Close.iloc[-len(Xs):,:]
    Close['SIGNAL'] = answer
    buy = Close[Close['SIGNAL']==1]['Close']
    sell = Close[Close['SIGNAL']==2]['Close']
    Close['Close'].plot()
    plt.scatter(list(buy.index),list(buy.values),color='red',marker="^")
    plt.scatter(list(sell.index),list(sell.values),color='green',marker='v')
    plt.show()
    
    return answer[-1]

def main():
    model = load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    df = yf.download('^TWII', start=start_date, end=end_date)
    signal = process(df,min_max_scaler,model)
    if signal == 0:
        send_to_telegram('不動作')
    if signal == 1:
        send_to_telegram('買進')
    if signal == 2:
        send_to_telegram('賣出')

if __name__ == '__main__':
    main()
    '''
    schedule.every().day.at('08:00').do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
    '''