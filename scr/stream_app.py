import streamlit as st
from dataload import dataload
from prediction import inference
from translate import translate_text
import time

# рандомный пример вопроса из базы
if 'example_input' not in st.session_state: 
    st.session_state.example_input = dataload().sample().values[0]

# словарь для переключения языков
translations = {
    "ru": {
        "sb_header": "Похожие медицинские вопросы",
        "sb_text": "Здесь вы можете найти похожие вопросы на медицинскую тематику",
        "qty_questions": "Как много похожих вопросов выводить?",
        "qty_questions_text": "Количество вопросов:",
        "input_text": "Здесь Ваш вопрос:",
        "button_text": "Поиск",
        "spinner_text": "Ищем...",
        "searching_time": "Время поиска, сек:",
        "result_text": "Ваш вопрос:",
    },

    "en": {
        "sb_header": "Similar medical questions",
        "sb_text": "Here you can find similar questions on the medical topics",
        "qty_questions": "How many similar questions to output?",
        "qty_questions_text": "Qty questions:",
        "input_text": "Here is your question:",
        "button_text": "Search",
        "spinner_text": "Searching...",
        "searching_time": "Searching time, sec:",
        "result_text": "Your question:",
    }
}

# инициализация состояния языка по умолчанию
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# кнопка для выбора языка
default_translate = 0
translate = st.radio("Language / Язык",
                     options=['English', 'Русский'],
                     index=default_translate,
                     horizontal=True)

if translate == 'Русский':
    st.session_state.language = 'ru'
    source_language = 'en'
else:
    st.session_state.language = 'en'
    source_language = 'ru'

# выбор языка из словаря
session_language = st.session_state.language
texts = translations[session_language]

# боковая панель с заголовками
with st.sidebar:
    st.header(texts['sb_header'])
    st.write(texts['sb_text'])

# кнопка для выбора количества схожих вопросов
min_val = 1
max_val = 10
default_slider = max_val 
val = st.slider(texts['qty_questions'], min_value=min_val, max_value=max_val, value=default_slider)
st.write(texts['qty_questions_text'], val)

# кнопка для ввода вопроса, текст по умолчанию - рандомный вопрос из базы, можно вводить свой вопрос на русском или английском
if session_language == 'ru':
    example_text = translate_text(st.session_state.example_input, source=source_language, target=session_language)
    user_input_ru = st.text_area(texts['input_text'], example_text,  height=100)
    user_input = translate_text(user_input_ru, source='ru', target='en')
else:
    example_text = st.session_state.example_input
    user_input_en = st.text_area(texts['input_text'], example_text, height=100)
    user_input = translate_text(user_input_en, source='ru', target='en')
    
st.session_state.saved_input = user_input

# кнопка для поиска схожих вопросов и спиннер для отображения поиска
col1, col2 = st.columns(2)

with col1:
    click = st.button(texts['button_text'])

if click:
# запуск поиска и расчет времени затраченного на обработку запроса    
    start_time = time.time()
    with col2:
        with st.spinner(texts['spinner_text']):
            translated_question, result_df = inference(st.session_state.saved_input, max_val, source='en', target='ru')
    end_time = time.time()
    processing_time = end_time - start_time
    
    st.write(texts['searching_time'], f'{processing_time:.2f}')

# результаты поиска в виде таблицы  
    if session_language == 'ru':
        st.write(texts['result_text'], user_input_ru)
        st.write(result_df.drop(columns='questions').rename(columns={'translated_questions': 'вопросы', 'cos_sim': 'сходство'}, inplace=False).head(val).to_markdown(index=False))
    else:
        st.write(texts['result_text'], user_input_en)
        st.write(result_df.head(val).drop(columns='translated_questions').to_markdown(index=False))