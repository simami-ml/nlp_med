import streamlit as st
from dataload import dataload
from prediction import inference
import pandas as pd
import time

with st.sidebar:
    st.header('Similar medical questions')
    st.write('Here you can find similar questions on the medical topics')

default_translate = 1
translate = st.radio('Need to translate?', options=['Yes','No'], index=default_translate)
st.write("You selected:", translate)

default_slider = 5  
val = st.slider('How many similar questions to output?', min_value=1, max_value=10, value=default_slider)
st.write('Qty questions:', val)

if 'saved_input' not in st.session_state:
    st.session_state.saved_input = dataload().sample().values[0]
user_input = st.text_area('Here is your question:', st.session_state.saved_input, height=100)
st.session_state.saved_input = user_input

col1, col2 = st.columns(2)

with col1:
    click = st.button('Search')

if click:
    start_time = time.time()
    if 'result_df' not in st.session_state:
        with col2:
            with st.spinner('Searching...'):
                translated_question, result_df = inference(st.session_state.saved_input, val)
                st.session_state.result_df = result_df
                st.session_state.translated_question = translated_question
    end_time = time.time()
    processing_time = end_time - start_time
    
    st.write(f'Searching time: {processing_time:.2f} seconds')
    st.write('Your question:', st.session_state.saved_input)

    if translate == 'Yes':
        st.write('Translated question:', st.session_state.translated_question)
        st.write(st.session_state.result_df.head(val).to_markdown(index=False))
    else:
        st.write(st.session_state.result_df.head(val).drop(columns='translated_questions').to_markdown(index=False))
          
    
