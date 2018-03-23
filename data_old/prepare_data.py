#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# Импортируем библиотеку для работы со временем
from datetime import datetime
# Импортируем библиотеку для работы с файлами csv
import csv


# Взять данные о курсе валют и перевести их в список
with open("DISSER-DATA.csv", "r") as file:
    # Данные о курсе валют лежат в файле DISSER-DATA.csv, программа считывает файл
    lines = file.readlines()
    # Программа удаляет заголовки таблиц, так как они не нужны
    data_raw = lines[1:]
    data = []
    for line in data_raw:
        # В каждой строчке файла, программа убирает техническую пунктуацию и делит строки на слова
        data.append(line.replace("\n", '').split("\t"))
# На выходе их этой части алгоритма мы получаем список из списков с данными о курсе валют с их датой

# Взять данные о курсе BRENT и перевести их в список
with open("DISSER-BRENT.csv", "r") as file2:
    lines = file2.readlines()
    data_raw = lines[1:]
    data2 = []
    for line in data_raw:
        data2.append(line.replace("\n", '').split("\t"))
# В этой части программы, система проводит те же операции, что и с курсом валют

# Все данные вставляем в одну таблицу
data_all = []
# Эта часть алгоритма, который заполянет пустые ячейки данных курса нефти, ее старыми значениями
for d in data:
    for d2 in reversed(data2):
        date1 = d[0].split(".")
        date2 = d2[0].split('.')
        # Эта часть кода заполняет таблицу текущими данными о курсе нефти
        # Создаем переменные с временем каждой таблицы
        date_of_kurs = datetime(int(date1[2]), int(date1[1]), int(date1[0]))
        date_of_neft = datetime(int(date2[2]), int(date2[1]), int(date2[0]))
        # Сравниваем время, для того чтобы определить курс нефти к курсу валют
        if date_of_kurs >= date_of_neft:
            # Программа переводит дату в удобную нам форму
            date = datetime(int(date1[2]), int(date1[1]), int(date1[0]))
            # И создает список с этими данными
            some = [date, float(d[1]), float(d[2]), float(d2[1])]
    # Создает список из списков данных
    data_all.append(some)

# Создаем заголовки таблицы, то есть дата, курс к доллару США, курс к рублю и курс нефти
headers = [["date", "USD", "RUB", "BRENT"]]

# Теперь сохраняем в таблице csv для чтения pandas
with open("data.csv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    headers.extend(reversed(data_all))
    writer.writerows(headers)

