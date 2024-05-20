from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from dataload import dataload
import warnings
warnings.filterwarnings(action="ignore")

def translate_text(text):
    MAX_LENGTH = 100
    NUM_BEAMS = 3
    EARLY_STOP = True

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_translate = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    sentences = sent_tokenize(text)
    translated_sentences = []
    for sentence in sentences:
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model_translate.generate(inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=EARLY_STOP)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_sentences.append(translated_text)
    translated_text = " ".join(translated_sentences)
    
    return translated_text


if __name__ == '__main__':
    sample_text = dataload().sample().values[0]

    print(sample_text, translate_text(sample_text))