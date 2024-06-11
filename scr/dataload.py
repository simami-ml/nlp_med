import os
import pandas as pd
from datasets import load_dataset

def dataload():

    dataset = load_dataset("medical_questions_pairs")
    df = dataset["train"].to_pandas()
    questions = pd.concat([df['question_1'],df['question_2']],axis=0).drop_duplicates(keep='first')
    return questions


if __name__ == '__main__':

    print(dataload().head())
