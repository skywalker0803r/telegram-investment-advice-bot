#此份檔案程式碼主要架構參考自chatgpt:https://sharegpt.com/c/kZOYTX1
'''
幫我寫一個程式功能如下:
有一個函數叫send_to_telegram(message)用來發送message
有一個函數叫process(df,min_max_scaler,model)用來根據輸入的參數產生交易信號
有一個函數叫main()此做幾件事情:
1.載入model.h5模型
2.載入scaler.pkl
3.使用yfinance取得最近180天^TWII資料稱為df
4.套用process(df,min_max_scaler,model)取得signal
5.當signal == 0,send_to_telegram('不動作')
6.當signal == 1,send_to_telegram('買進')
7.當signal == 2,send_to_telegram('賣出')

使用schedule套件讓這個程式每天早上8點執行一次
'''
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


# 發電報函數
def send_to_telegram(message):
    apiToken = '5850662274:AAGeKZqM1JfQfh3CrSKG6BZ9pEvDajdBUqs'
    chatID = '1567262377'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

# 計算金融技術指標函數
def calculate_ta(df):
    ta_functions = [TA.RSI, TA.WILLIAMS, TA.SMA, TA.EMA, TA.WMA, TA.HMA, TA.TEMA, TA.CCI, TA.CMO, TA.MACD, TA.PPO, TA.ROC, TA.CFI, TA.DMI, TA.SAR]
    ta_names = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI', 'DMI', 'SAR']
    for i, ta_func in enumerate(ta_functions):
        try:
            df[ta_names[i]] = ta_func(df)
        except:
            if ta_names[i] == 'MACD':
                df[ta_names[i]] = ta_func(df)['MACD']-ta_func(df)['SIGNAL']
            if ta_names[i] == 'PPO':
                df[ta_names[i]] = ta_func(df)['PPO']-ta_func(df)['SIGNAL']  
            if ta_names[i] == 'DMI':
                df[ta_names[i]] = ta_func(df)['DI+']-ta_func(df)['DI-']
    return df

# 給定歷史數據,scaler,model,輸出買賣訊號的函數
def process(df,min_max_scaler,model):
    df = calculate_ta(df)
    df = df.dropna(axis=0)
    features = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI', 'DMI', 'SAR']
    Close = df[['Close']]
    df = df[features]
    # 數據轉換
    df[features] = min_max_scaler.transform(df[features])
    # 製作X(15x15's array)
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
    # 模型預測買賣訊號
    singal = model.predict(Xs)
    singal = [ np.argmax(i) for i in singal]
    
    # 繪圖
    Close = Close.iloc[-len(Xs):,:]
    Close['SIGNAL'] = singal
    buy = Close[Close['SIGNAL']==1]['Close']
    sell = Close[Close['SIGNAL']==2]['Close']
    
    Close['Close'].plot()
    plt.scatter(list(buy.index),list(buy.values),color='red',marker="^")
    plt.show()
    
    Close['Close'].plot()
    plt.scatter(list(sell.index),list(sell.values),color='green',marker='v')
    plt.show()
    
    return singal[-1]

def main():
    # 載入模型和scaler
    model = load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    # 資料時間範圍設定
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    symbol = '^TWII'
    df = yf.download(symbol, start=start_date, end=end_date)
    # 取得交易訊號
    signal = process(df,min_max_scaler,model)
    if signal == 0:
        action = '不動作'
    if signal == 1:
        action = '建議買入'
    if signal == 2:
        action = '建議賣出'
    # 發送消息至telegram
    current_time = dt.datetime.now().strftime("%Y-%m-%d")
    message = f"現在時間:{current_time} 投資項目:{symbol} 當前建議操作:{action}"
    send_to_telegram(message)

if __name__ == '__main__':
    main()
    schedule.every().day.at('08:00').do(main)#每日八點準時執行
    while True:
        schedule.run_pending()
        time.sleep(1)
