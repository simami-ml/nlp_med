from dataload import dataload
from deep_translator import GoogleTranslator

def translate_text(text, source, target):
    translator = GoogleTranslator(source=source, target=target)
    translated_text = translator.translate(text)
    
    return translated_text


if __name__ == '__main__':
    sample_text = dataload().sample().values[0]

    print(sample_text, translate_text(sample_text, 'en', 'ru'))