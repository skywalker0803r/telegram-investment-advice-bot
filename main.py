import yfinance as yf
import telegram
from finta import TA

# 設定Telegram機器人的TOKEN和聊天室ID
bot_token = 'your_bot_token'
chat_id = 'your_chat_id'

def process(df,min_max_scaler,model):
    # 計算技術指標特徵
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
    df = df[features]
    
    # minmax scaler
    df = min_max_scaler.transform(df)
    
    # 製作Xs
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
    ys = model.predict(Xs)
    signal = np.argmax(ys[-1])
    
    # 返回訊號
    return signal

def send_to_telegram(message):
    # 初始化Telegram機器人
    bot = telegram.Bot(token=bot_token)
    # 傳送訊息到指定聊天室
    bot.send_message(chat_id=chat_id, text=message)

def main():
    # keras載入模型
    'model = '
    
    # 載入minmaxscaler
    'minmaxscaler = '
    
    # 使用yf.download和 symbol 取得近期60天'^TWII'的資料稱之為df
    'df = '
    
    # 將資料傳入預處理函式，得到買賣訊號
    signal = process(df,min_max_scaler,model)
    
    # 根據買賣訊號發送Telegram訊息
    if signal == 0:
        send_to_telegram('不動作')
    elif signal == 1:
        send_to_telegram('買進')
    else:
        send_to_telegram('賣出')

if __name__ == '__main__':
    # 設定每天早上8點執行main()函式
    schedule.every().day.at('08:00').do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
