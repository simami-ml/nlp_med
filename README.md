**Инженер машинного обучения.**
# **Проект**

У компании, которая развивает сеть тематических
форумов дела идут не очень – пользователи просматривают всего лишь
несколько страниц и уходят на ресурсы конкурентов. Расходы на
закупку трафика не окупают себя.

**Гипотеза:** давайте показывать пользователям похожие вопросы, чтобы
удерживать пользователей и растить просмотры.

**Цель проекта:** разработать сервис для поиска похожих вопросов.

**Исходные данные:** набор пар медицинских вопросов с указанием, похожи ли они между собой.<br>
[Датасет](https://huggingface.co/datasets/medical_questions_pairs "клац") 

В ходе проекта:
- изучили датасет
- обработали текстовые данные
- нашли способ подобрать близкие по смыслу вопросы
- собрали baseline решение
- сделали демо приложение Streamlit

Работа над данными проведена в ноутбуке -> [посмотреть notebook](https://github.com/simami-ml/nlp_med/blob/main/notebooks/medical_questions.ipynb 'клац')

Папка со скриптами python -> [посмотреть код](https://github.com/simami-ml/nlp_med/tree/main/scr 'клац')

Сервис на Streamlit -> [протестировать сервис](https://nlpmed-vbkkfcurgtswkusmh6bagg.streamlit.app 'клац') 