import pandas as pd

def dataload():
    df = pd.read_csv(f'/mount/src/nlp_med/data/train_df_processed.csv', index_col=[0])
    questions = pd.concat([df['question_1'],df['question_2']],axis=0).drop_duplicates(keep='first')
    return questions


if __name__ == '__main__':

    print(dataload().head())
