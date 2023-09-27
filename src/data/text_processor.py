import re
import spacy
import pandas as pd
from tqdm import tqdm


class TextPreprocessor:
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(self, texts, lowercase=True, remove_punct=True, 
                   remove_num=True, remove_stop=True, lemmatize=True):
        return [self.preprocess_text(str(t), lowercase, remove_punct, remove_num, remove_stop, lemmatize) for t in tqdm(texts, desc='Pre-processing')]

    def preprocess_text(self, text, lowercase, remove_punct,
                        remove_num, remove_stop, lemmatize):
        if lowercase:
            text = self._lowercase(text)
        doc = self.nlp(text)
        if remove_punct:
            doc = self._remove_punctuation(doc)
        if remove_num:
            doc = self._remove_numbers(doc)
        if remove_stop:
            doc = self._remove_stop_words(doc)
        if lemmatize:
            text = self._lemmatize(doc)
        else:
            text = self._get_text(doc)
        return text

    def _lowercase(self, text):
        return text.lower()
    
    def _remove_punctuation(self, doc):
        return [t for t in doc if not t.is_punct]
    
    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.is_digit or t.like_num or re.match('.*\d+', t.text))]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _get_text(self, doc):
        return ' '.join([t.text for t in doc])
