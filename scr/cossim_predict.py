import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings(action="ignore")

def translate_text(text):
    MAX_LENGTH = 100
    NUM_BEAMS = 3
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_translate = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model_translate.generate(inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def df():
    df = pd.read_csv(f'../data/train_df_processed.csv', index_col=[0])
    questions = pd.concat([df['question_1'],df['question_2']],axis=0).drop_duplicates(keep='first')
    return questions


def model_m(question):

    model_mlm = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    questions = df()
    questions_embeddings = model_mlm.encode(questions.tolist())
    
    question_embedding = model_mlm.encode(question)
    
    predict = np.array([util.cos_sim(questions_embeddings, question_embedding)])

    return predict


def main():
    questions = df()
    
    question = questions.sample().values[0]
    
    translated_question = translate_text(question)
    predict = model_m(question)
    N = 11
    values_top = np.argsort(predict.reshape(-1))[::-1][:N]
    res_df = pd.DataFrame(columns=['questions', 'translated_questions','cos_sim']) 
    
    for val in values_top:
        if questions.tolist()[val] == question:
            pass        
        else: 
            q = questions.tolist()[val]
            cos_sim = predict.reshape(-1)[val].round(1)
            translated_q = translate_text(q)
            res_df.loc[val] = [q,translated_q,cos_sim]

    print(question, translated_question, res_df.head())

if __name__ == '__main__':
    main()