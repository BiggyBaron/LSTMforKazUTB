#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Скачиваем веб-фреймворк Flask
from flask import Flask, render_template
# Скачиваем наш API
import download_API
# Скачиваем нашу ИНС
import main
# Скачиваем библиотеку для работы со временем
from datetime import datetime
# Скачиваем библиотеку для работы с базой данных
import csv


# Создаем веб-приложение Flask
app = Flask(__name__)

# Создаем страницу интерфейса на значении без домена
@app.route('/')
def hello():
    # Скачиваем курсы валют и нефти через наш API
    USD, RUB = download_API.courses()
    BRENT = download_API.brent()
    # Осознаем какая сейчас дата
    date = datetime.now()
    # Теперь сохраняем в базу данных
    with open("./data/data.csv", 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows([[date, USD, RUB, BRENT]])
    # Запускаем ИНС для расчета курса завтрашнего дня тенге к доллару США
    KURS = str(main.gui())
    # Рендер страницы с отправлением локальных переменных
    return render_template('main.html', **locals())


# Запускаем веб-интерфейс
if __name__ == "__main__":
    # На адрес localhost с портом 8090
    app.run(host='0.0.0.0', port=8090, debug=True)

