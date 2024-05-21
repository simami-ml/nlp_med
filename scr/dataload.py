import pandas as pd

def dataload():
    try:
        df = pd.read_csv(f'../data/train_df.csv', index_col=[0])
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/simami-ml/nlp_med/main/data/train_df.csv', index_col=[0])
   
    questions = pd.concat([df['question_1'],df['question_2']],axis=0).drop_duplicates(keep='first')
    return questions


if __name__ == '__main__':

    print(dataload().head())
