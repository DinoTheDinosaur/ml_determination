import re
import spacy

import numpy as np

from collections import Counter


class MatrixLanguageDeterminerP12:
    def __init__(self, device='cpu', L1=None, L2=None, config={}, alpha=0):
        from ml_determination.get_monolingual_translations import Translator
        from ml_determination.get_segment_scores import Scorer
        self.nmt = Translator(L1=L1, L2=L2)
        self.scorers = {
            L1: Scorer(data_path=config[L1]['data_path'], model_path=config[L1]['model_path'], device=device),
            L2: Scorer(data_path=config[L2]['data_path'], model_path=config[L2]['model_path'], device=device)
        }
        self.L1 = L1
        self.L2 = L2
        self.alpha = alpha

    def compute_likelihoods(self, sentence, lang_ids):
        try:
            result = self.nmt.translate(sentence, lang_ids)
        except ValueError:
            return
        scores = {}
        for lang in result:
            scores[lang] = self.scorers[lang].compute_scores(result[lang])
        return scores

    def sentence_likelihoods(self, sentence, lang_ids):
        likelihoods = self.compute_likelihoods(sentence, lang_ids)
        out = {self.L1: None, self.L2: None}
        if likelihoods != None:
            out[self.L1] = 0 - np.sum(likelihoods[self.L1])
            out[self.L2] = 0 - np.sum(likelihoods[self.L2])
        return out

    def determine_ML(self, sentence, lang_ids=None):
        likelihoods = self.sentence_likelihoods(sentence, lang_ids)
        if None not in likelihoods.values():
            diff = likelihoods[self.L2] - likelihoods[self.L1]
            ml = self.L1 if diff < self.alpha else self.L2
            return ml
        else:
            return None

    def determine_ML_div(self, sentence, lang_ids=None):
        likelihoods = self.sentence_likelihoods(sentence, lang_ids)
        if None not in likelihoods.values():
            diff = likelihoods[self.L2] / likelihoods[self.L1]
            ml = self.L1 if diff < self.alpha else self.L2
            return ml
        else:
            return None

class MatrixLanguageDeterminerP11:
    def __init__(self, L1, L2):
        self.L1 = L1
        self.L2 = L2
        self.patterns_segments = {
            'ZH': re.compile(r'[\u4e00-\u9fff]+'),
            'EN': re.compile(r'[a-z\-\']+')
        }

    def determine_ML(self, sentence, lang_ids=''):
        has_big_segs = {}
        if lang_ids == '':
            for word in sentence.split():
                matched = False
                for lang in self.patterns_segments:
                    if re.match(self.patterns_segments[lang], word):
                        lang_ids += lang + ' '
                        matched = True
                if not matched:
                    lang_ids += 'UNK '
        else:
            lang_ids += ' '
        for ml in [self.L1, self.L2]:
            if ml not in lang_ids:
                return
            el = list({self.L1, self.L2} - {ml})[0]
            matches = re.findall(
                f'{ml} {ml} ', lang_ids
            ) + re.findall(
                f'<s> {ml} {el} {ml} <s>', '<s> ' + lang_ids + ' <s>'
            )
            has_big_segs[ml] = False if matches == [] else True
        if np.sum(np.sum(list(has_big_segs.values()))) == 2:
            return
        for lang in [self.L1, self.L2]:
            if has_big_segs[lang]:
                return lang

class MatrixLanguageDeterminerP2:
    def __init__(self, L1, L2, pos_whitelist={
        'EN': ['DET', 'AUX', 'SCONJ', 'CCONJ'],
        'ZH': ['DET', 'AUX', 'SCONJ', 'CCONJ'],
        'ES': ['DET', 'AUX', 'SCONJ', 'CCONJ']
    }):
        self.L1 = L1
        self.L2 = L2
        self.patterns_segments = {
            'ZH': re.compile(r'[\u4e00-\u9fff]+'),
            'EN': re.compile(r'[a-z\-\']+')
        }
        self.pos_whitelist = pos_whitelist
        self.nlp = {}
        for lang in [L1, L2]:
            if lang == 'ES':
                self.nlp[lang] = spacy.load(f"{lang.lower()}_core_news_md", exclude=['ner'])
            else:
                self.nlp[lang] = spacy.load(f"{lang.lower()}_core_web_sm", exclude=['ner'])

    def determine_ML(self, sentence, lang_ids=''):
        if lang_ids == '':
            for word in sentence.split():
                matched = False
                for lang in self.patterns_segments:
                    if re.match(self.patterns_segments[lang], word):
                        lang_ids += lang + ' '
                        matched = True
                if not matched:
                    lang_ids += 'UNK '
        else:
            lang_ids += ' '
        has_system = dict(zip(
            [self.L1, self.L2],
            [False]*2
        ))
        pat = re.compile(f'(({self.L1} )+|({self.L2} )+)')
        islands = re.findall(pat, lang_ids + ' ')
        sentence = sentence.split()
        i = 0
        for island in islands:
            lid_tokens, L1_matches, L2_matches = list(item.strip() for item in island)
            lid = self.L1 if L1_matches else self.L2
            j = len(lid_tokens.split())
            for token in self.nlp[lid](' '.join(sentence[i:j])):
                if token.pos_ in self.pos_whitelist[lid]:
                    has_system[lid] = True
        if np.sum(list(has_system.values())) == 2:
            return
        for lang in has_system:
            if has_system[lang]:
                return lang
        return

class MatrixLanguageDeterminerWordMajority:
    def __init__(self, L1, L2):
        self.L1 = L1
        self.L2 = L2
        self.patterns_segments = {
            'ZH': re.compile(r'[\u4e00-\u9fff]+'),
            'EN': re.compile(r'[a-z\-\']+')
        }

    def determine_ML(self, sentence, lang_ids=''):
        if lang_ids == '':
            for word in sentence.split():
                matched = False
                for lang in self.patterns_segments:
                    if re.match(self.patterns_segments[lang], word):
                        lang_ids += lang + ' '
                        matched = True
                if not matched:
                    lang_ids += 'UNK '
        lang_counts = Counter(lang_ids.strip().split())
        max_lang = None
        for lang in [self.L1, self.L2]:
            if max_lang == None:
                max_lang = lang
            elif lang_counts[lang] > lang_counts[max_lang]:
                max_lang = lang
            elif lang_counts[lang] == lang_counts[max_lang]:
                return
        return max_lang