from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataload import dataload
import warnings
warnings.filterwarnings(action="ignore")


def translate_text(text):
    MAX_LENGTH = 100
    NUM_BEAMS = 3
    EARLY_STOP = True

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_translate = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model_translate.generate(inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=EARLY_STOP)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text


if __name__ == '__main__':

    print(translate_text(dataload().sample().values[0]))