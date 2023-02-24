import yfinance as yf
import telegram

# 設定Telegram機器人的TOKEN和聊天室ID
bot_token = 'your_bot_token'
chat_id = 'your_chat_id'

def preprocess(df):
    # 在這裡加上您的預處理代碼
    # 這個例子中我們假設只要收盤價高於開盤價就買進，反之就賣出
    if df['Close'] > df['Open']:
        return 1
    elif df['Close'] < df['Open']:
        return -1
    else:
        return 0

def send_to_telegram(message):
    # 初始化Telegram機器人
    bot = telegram.Bot(token=bot_token)
    # 傳送訊息到指定聊天室
    bot.send_message(chat_id=chat_id, text=message)

def main():
    # 取得蘋果股票的資料
    symbol = 'AAPL'
    data = yf.download(symbol, start='2023-01-01', end='2023-02-23')
    # 將資料傳入預處理函式，得到買賣訊號
    signal = preprocess(data.tail(1))
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
