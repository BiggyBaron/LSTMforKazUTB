## Добро пожаловать на проект "Разработка нейросетевого программного робота для анализа тенденций на торговых биржах"
Проект создан для диссертации по защите степени в КазУТБ.
![Вот он](http://www.kazutb.kz/ru/cache/widgetkit/gallery/3/5-39db279268.jpg)
## Установка
Дальше идет инструкция по установке проекта, но она с учетом того, что Вы знаете хотя бы основы работы на Линуксе и Питоне.
### Пререквизиты
Необходимые составляющие проекта:
1. Работающий компьютер с системой Linux (Лучше всего Debian или Ubuntu)
2. Ровные ручки
3. Интернет
4. Клиент GIT, [тут](https://git-scm.com/book/ru/v1/%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5-%D0%A3%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-Git) лежит инструкция как ее установить

#### Скачивание проекта

Тут все просто:
1. Откройте терминал.
2. Запускайте этот скрипт:
```bash
git clone https://github.com/BiggyBaron/LSTMforKazUTB.git
```
3. Зайдите в папку проекта, и сидите там, так как еще много чего надо будет сделать.
```bash
cd LSTMforKazUTB
```

#### Установка среды Anaconda
![Картинка Анаконды](https://binstar-static-prod.s3.amazonaws.com/latest/img/AnacondaCloud_logo_green.png)
[Anaconda](https://anaconda.org/) - специальная среда Python для специалистов больших данных, именно нам она нужна.
** Далее инструкция только для ОС Debian**
1. [Вот тут](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) лежит инструкция по установке Anaconda.
2. Теперь надо создать виртуальную среду из файла (вы должны быть в папке проекта):
```bash
conda env create -f ml.yml
```
3. Теперь активируйте среду:
```bash
source activate ml
```
4. Запускайте страницу и любуйтесь:
```bash
python3 web-page.py
```
5. Заходите в браузере по адресу: http://localhost:8090/
6. ???
7. PROFIT!

### Файлы в проекте
Далее представлены файлы и папки проекта, чтобы Вы не запутались.

* LSTMforKazUTB/ - Скачанная Вами папка
* ├── data - Папка с базой данных
* │   ├── data.csv - База данных для анализа, которая создается программой
* │   ├── DISSER-BRENT.csv - Скачанная с сайта Investing.com база данных о цене нефти с сентября 2015 г бренда BRENT
* │   ├── DISSER-DATA.csv - Скачанная с официального сайта Национального банка РК курсы доллара и рубля
* │   └── prepare_data.py - Модуль, который создает базу данных в один файл
* ├── download_API.py - API, который скачивает значения курса валют и нефти BRENT
* ├── LICENSE - Лицензия MIT
* ├── main.py - ПО ИНС, тренировка и использование ИНС
* ├── README.md - Файл "ПРОЧИТАЙ МЕНЯ"
* ├── requirements.txt - Файл со списком необходимых библиотек Python, если не хотите делать conda
* ├── static - Файлы для работы сайта
* │   └── ActualvsPredicted.png - Картинка график предугаданных значений
* ├── templates - Папка с html файлом сайта
* │   └── main.html - Сам сайт
* ├── web-page.py - Скрипт, который запускает веб-интерфейс
* ├── ml.yml - Виртуальная среда этого проекта
* ├── _config.yml - Забей, это для Github нужно
* ├── сохранение_ИНС.json - Сохраненная модель ИНС
* └── сохранение_ИНС_весы.h5 - Весы модели ИНС

### Как это работает?
#### Подготовка данных
Подготовка данных идет в модуле prepare_data.py.
1. Необходимо скачать данные о курсе из официального сайта Национального Банка [тут](http://nationalbank.kz/?docid=748&switch=russian)
2. ** Качаем только доллар США и рубль**  с 1 сентября 2015 по нынешнюю дату
3. Упаковываем в файл DISSER-DATA.csv, _проверьте чтобы было похоже на то, что было_
4. Заходим на сайт Investing.com и регистрируемся [тут](https://ru.investing.com/commodities/brent-oil-historical-data)
5. Качаем данные с 1 сентября 2015 года по нынешнюю дату
6. Сохраняем в файл DISSER-BRENT.csv, ** данные должны выглядеть также как и в оригинале документа**
7. Запускаем скрипт:
```bash
python3 ./data/prepare_data.py
```
8. Открываем файл data.csv
9. Проверяем данные

#### Обучение нейронной сети
1. Запускаем скрипт:
```bash
python3 main.py
```
2. Смотрим как раз за разом появляются графики и заумные выводы в терминал
3. Круто! Теперь у Вас есть ИНС.

#### Запуск самой системы
1. Запускайте страницу и любуйтесь:
```bash
python3 web-page.py
```
2. Заходите в браузере по адресу: http://localhost:8090/
3. ???
4. PROFIT!

#### Больше информации
В ближайшее время замучу Вики страницу, там выложу больше информации.
А может просто залью диссер сюда, я там как для идиотов писал, каждый пункт.

#### Об авторе
Я крут.
