#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Библиотека для скачивания данных с сайта
import urllib.request
# Библиотека для перевода данных в json и их парсинг
import json


# Узнать данные о курсе тенге к доллару
def courses():
    # Открываем данные из data.fixer
    with urllib.request.urlopen('http://data.fixer.io/api/latest?access_key='
                                '7ef027c87c9f00b3c0299ea32aba387e') as response:
        html = response.read()
        # Переводим их в json
        cont = json.loads(html.decode('utf-8'))
        # Парсим данные
        kzt = cont["rates"]["KZT"]
        usd = cont["rates"]["USD"]
        rub = cont["rates"]["RUB"]
        # Расчитываем курс тенге к доллару США и к рублю РФ
        usd_in_kzt = kzt/usd
        rub_in_kzt = kzt/rub
        # Возращаем данные
        return usd_in_kzt, rub_in_kzt


# Узнать данные о цене Brent
def brent():
    # Скачиваем данные с сайта quandl
    with urllib.request.urlopen('https://www.quandl.com/api/v3/datasets/CHRIS/ICE_B1/data.json') as response:
        html = response.read()
        # Переводим их в json
        cont = json.loads(html.decode('utf-8'))
        # Находим значение цены Brent
        price_settle = cont["dataset_data"]["data"][0][4]
        # Возвращаем данные
        return price_settle

