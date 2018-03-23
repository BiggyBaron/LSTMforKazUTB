from datetime import datetime
import csv
from pandas import read_csv


# Взять данные о курсе валют и перевести их в список
with open("DISSER-DATA.csv", "r") as file:
    lines = file.readlines()
    data_raw = lines[1:]
    data = []
    for line in data_raw:
        data.append(line.replace("\n", '').split("\t"))

# Взять данные о курсе BRENT и перевести их в список
with open("DISSER-BRENT.csv", "r") as file2:
    lines = file2.readlines()
    data_raw = lines[1:]
    data2 = []
    for line in data_raw:
        data2.append(line.replace("\n", '').split("\t"))

# Оба данные вставляем в одну таблицу
data_all = []
for d in data:
    for d2 in reversed(data2):
        date1 = d[0].split(".")
        date2 = d2[0].split('.')
        if datetime(int(date1[2]), int(date1[1]), int(date1[0])) >= datetime(int(date2[2]), int(date2[1]), int(date2[0])):
            date = datetime(int(date1[2]), int(date1[1]), int(date1[0]))
            some = [date, float(d[1]), float(d[2]), float(d2[1])]
    data_all.append(some)

headers = [["date", "USD", "RUB", "BRENT"]]
# Теперь сохраняем в таблице csv для чтения pandas
with open("data.csv", 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    headers.extend(reversed(data_all))
    writer.writerows(headers)

