import requests

def send_to_telegram(message):

    apiToken = '5850662274:AAGeKZqM1JfQfh3CrSKG6BZ9pEvDajdBUqs'
    chatID = '1567262377'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

send_to_telegram("Hello from Python!")
