import numpy as np
from sentence_transformers import SentenceTransformer, util
from dataload import dataload
import warnings
warnings.filterwarnings(action="ignore")

def model_m(question):

    model_mlm = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    questions = dataload()
    questions_embeddings = model_mlm.encode(questions.tolist())
    question_embedding = model_mlm.encode(question)
    predict = np.array([util.cos_sim(questions_embeddings, question_embedding)])
    
    return predict


if __name__ == '__main__':
   
   question = dataload().sample().values[0]
   print(model_m(question)[:5])