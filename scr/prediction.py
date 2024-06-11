import pandas as pd
import numpy as np
from dataload import dataload
from translate import translate_text
from model import model_m
import warnings
warnings.filterwarnings(action="ignore")

questions = dataload()

def inference(question):

    translated_question = translate_text(question)
    predict = model_m(question)

    values_top = np.argsort(predict.reshape(-1))[::-1]
    res_df = pd.DataFrame(columns=['questions', 'translated_questions','cos_sim']) 
    
    for val in values_top:
        if questions.tolist()[val] == question:
            pass        
        else: 
            q = questions.tolist()[val]
            cos_sim = predict.reshape(-1)[val].round(1)
            translated_q = translate_text(q)
            res_df.loc[val] = [q,translated_q,cos_sim]

    return translated_question, res_df.iloc[1:]

def main():
    question = dataload().sample().values[0]
    pred = inference(question)
    print(question, pred)

if __name__ == '__main__':
    main()