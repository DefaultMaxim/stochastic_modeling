import requests
from tinkoff.invest.utils import quotation_to_decimal
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TOKEN')

url = os.getenv('URL_ORDER')

# Ваш токен авторизации
TOKEN = os.getenv('TOKEN')

# Заголовки запроса
headers = {
    "Authorization": f"Bearer {TOKEN}"
}

# Выполнение GET-запроса
response = requests.post(
    url,
    headers=headers,
    json={"figi": "BBG004731032",
          "depth": 20,
          "instrumentId": "BBG004731032"
          }
)

# Проверка статуса ответа
if response.status_code == 200:
    # Успешный ответ
    data = response.json()  # Если ожидается JSON-ответ
else:
    # Обработка ошибки
    print(f"Ошибка: {response.status_code}")
    

def quot_to_float(
    quot: dict
) -> float:
    """
    Transform tinkoff.Quatation to float.

    Args:
        quot (dict): Dictionary {'units': int, 'nano': int}

    Returns:
        float: Transformed number.
    """

    return float(quot['units']) + float(quot['nano'] / (10**float(len(str(quot['nano'])))))


def transform_orderBook_data(
    data: dict,
    order_type: str,
    attr_type: str
) -> np.ndarray:
    """Transforms orderBook data to np.ndarray and returns prices or quantitys of bids or asks orders.

    Args:
        data (dict): Data from orderBook responce.
        order_type (str): 'bid' or 'ask' returns binds or asks.
        attr_type (str): 'price' or 'quant' returns prices or quantitys.

    Returns:
        np.ndarray: Prices or quantitys of bids or asks.
    """

    if order_type == 'bid':

        if attr_type == 'price':

            prices_bids = np.array([data['bids'][i]['price']
                                   for i in range(len(data['bids']))])
            prices_bids = np.array(
                list(map(lambda x: quot_to_float(x), prices_bids)))
            return prices_bids

        elif attr_type == 'quant':

            quantity_bids = np.array([float(data['bids'][i]['quantity'])
                                      for i in range(len(data['bids']))])
            return quantity_bids

        else:

            raise (AttributeError)

    elif order_type == 'ask':

        if attr_type == 'price':

            prices_asks = np.array([data['asks'][i]['price']
                                   for i in range(len(data['asks']))])
            prices_asks = np.array(
                list(map(lambda x: quot_to_float(x), prices_asks)))

            return prices_asks

        elif attr_type == 'quant':

            quantity_asks = np.array(
                [float(data['asks'][i]['quantity']) for i in range(len(data['asks']))])

            return quantity_asks

        else:

            raise (AttributeError)

    else:

        raise (AttributeError)


prices_bids = transform_orderBook_data(data, 'bid', 'price')
quantity_bids = transform_orderBook_data(data, 'bid', 'quant')

prices_asks = transform_orderBook_data(data, 'ask', 'price')
quantity_asks = transform_orderBook_data(data, 'ask', 'quant')

plt.hist(prices_asks, weights=quantity_asks, color='blue', label='asks')
plt.hist(prices_bids, weights=quantity_bids, color='red', label='bids')
plt.legend()
plt.grid()
plt.show()
