import re

from deep_translator import GoogleTranslator
from functools import partial
from nltk.stem import SnowballStemmer

class Translator:
    def __init__(self, L1=None, L2=None):
        self.L1 = L1
        self.L2 = L2
        self.nmt = {L1: {}, L2: {}}
        lang_codings = {'EN': 'english', 'ES': 'spanish', 'ZH': 'chinese (simplified)'}
        if ('EN' in [L1, L2]) and ('ZH' in [L1, L2]):
            zh_pat = re.compile(r'[\u4e00-\u9fff]+')
            with open('/share/mini1/usr/olga/projects/seame_fa/data/translation/en-cmn-enwiktionary.txt') as f:
                for line in f.read().split('\n'):
                    if line == '':
                        continue
                    if line[0] == '#':
                        continue
                    line_split = line.split('::')
                    if len(line_split) != 2:
                        continue
                    word = line.split('{')[0].strip().lower()
                    translations = list(re.search(zh_pat, translation).group(0) for translation in line_split[1].split(',') if re.search(zh_pat, translation) != None)
                    if translations == []:
                        continue
                    self.nmt['EN'][word] = translations
                    for translation in translations:
                        if translation not in self.nmt['ZH']:
                            self.nmt['ZH'][translation] = [word]
                        else:
                            self.nmt['ZH'][translation] += [word]
        if ('EN' in [L1, L2]) and ('ES' in [L1, L2]):
            es_pat = re.compile('[\w\s]+')
            with open('/share/mini1/usr/olga/projects/seame_fa/data/translation/en-es-enwiktionary.txt') as f:
                for line in f.read().split('\n'):
                    if line == '':
                        continue
                    if line[0] == '#':
                        continue
                    line_split = line.split('::')
                    if len(line_split) != 2:
                        continue
                    word = line.split(',')[0].strip().lower()
                    translations = list(re.search(es_pat, translation).group(0).lower() for translation in line_split[1].split(',') if re.search(es_pat, translation) != None)
                    if translations == []:
                        continue
                    self.nmt['EN'][word] = translations
                    for translation in translations:
                        if translation not in self.nmt['ES']:
                            self.nmt['ES'][translation] = [word]
                        else:
                            self.nmt['ES'][translation] += [word]

        self.nmt_g = {
            L1: GoogleTranslator(source=lang_codings[L1], target=lang_codings[L2]),
            L2: GoogleTranslator(source=lang_codings[L2], target=lang_codings[L1])
        }
        self.extract_subwords = {}
        for lang in [L1,L2]:
            if lang == 'ZH':
                func = lambda word: ' _'.join(list(word))
                self.extract_subwords[lang] = func
            else:
                self.extract_subwords[lang] = partial(self.extract_subwords_stem, ps=SnowballStemmer(lang_codings[lang]))
        

    def is_english(self, c):
        """check character is in English"""
        return ord(c.lower()) >= ord("a") and ord(c.lower()) <= 255


    def is_mandarin(self, c):
        """check character is Mandarin"""
        return (
            not self.is_english(c)
            and not c.isdigit()
            and c != " "
            and c != "<"
            and c != ">"
            and c != "'"
        )

    def extract_mandarin_only(self, text):
        """remove other symbols except for Mandarin characters in a string"""
        return "".join([c for c in text if self.is_mandarin(c)])

    def extract_non_mandarin(self, text):
        """remove Mandarin characters in a string"""
        return " ".join([w for w in text.split(" ") if not any(self.is_mandarin(c) for c in w)])

    def determine_language(self, word):
        man_match = self.extract_mandarin_only(word).strip()
        en_match = self.extract_non_mandarin(word).strip()
        if len(man_match) > 0:
            if len(en_match) > 0:
                warnings.warn(f'Word {word} is being weird!!!')
                return
            else:
                return 'ZH'
        else:
            return 'EN'

    def extract_subwords_stem(self, word, ps=None):
        stem = ps.stem(word)
        if stem != word and (stem in word):
            return stem + ' _' + word[len(stem):]
        return word

    def translate_words(self, sentence_split, lang_ids):
        result = []
        for word, orig_lang in zip(sentence_split, lang_ids):
            if len(word.strip()) == 0:
                continue
            if orig_lang == 'UNK':
                result += [{
                    self.L1: [word],
                    self.L2: [word]
                }]
            else:
                if not orig_lang:
                    orig_lang = self.determine_language(word)
                target_lang = self.L2 if orig_lang == self.L1 else self.L1
                if word not in self.nmt[orig_lang]:
                    translation = self.nmt_g[orig_lang].translate(word)
                    if translation:
                        synonyms = [translation.lower()]
                        self.nmt[orig_lang][word] = synonyms
                else:
                    synonyms = self.nmt[orig_lang][word]
                    if target_lang == 'ZH':
                        synonyms = list(self.extract_mandarin_only(synonym) for synonym in synonyms)
                result += [{
                    orig_lang: [self.extract_subwords[orig_lang](word)],
                    target_lang: list(self.extract_subwords[target_lang](synonym) for synonym in synonyms)
                }]
        return result

    def translate(self, sentence, lang_ids=None):
        sentence_split = sentence.split()
        if lang_ids:
            lang_ids = lang_ids.split()
        else:
            lang_ids = [None] * len(sentence_split)
        synonym_list = self.translate_words(sentence_split, lang_ids)
        result = {}

        for lang in [self.L1, self.L2]:
            original_result = []
            for word_variants in synonym_list:
                original_result += [word_variants[lang][0]]
            result[lang] = ' '.join(original_result)
        return result