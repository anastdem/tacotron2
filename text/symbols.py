""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_pad = '_'
# _punctuation = '!\'(),.:;? '
# _special = '-'

# вариант с ударением
# _punctuation = '!\'(),.:;?- '
# _special = '+'

_punctuation = '!\'"(),.:;?+ '
_special = '-'

_letters_en = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_rus = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

_phonems = ['sil', 'spn', 'a', 'ɐ', 'mʲ', 'pʲ', 'e', 'r', 'ə', 'k', 'm', 'ɪ', 'x', 'c', 'f', 'ʊ', 'ɨ', 'n̪', 'ɟ', 'tɕ',
            'j', 'o', 'vʲ', 't̪s̪', 's̪', 'ɕː', 'ɲ', 'ç', 'p', 'b', 'sʲ', 'd̪', 'd̪ː', 'ʐ', 'u', 'rʲ', 'z̪', 'zʲ', 'i', 't̪',
            'tʲ', 'ʎ', 'ɫ', 'v', 'ɡ', 'ʂ', 'ɛ', 'bː', 'bʲː', 'bʲ', 'dʲ', 'dʐː', 'ɫː', 'rː', 'n̪ː', 'sʲː', 't̪s̪ː', 'fʲ',
            's̪ː', 'ʉ', 'vː', 'vʲː', 'pː', 'pʲː', 'ɵ', 'æ', 'ɲː', 'tɕː', 'ɣ', 'ɟː', 'dʲː', 'kː', 'cː', 'ʎː', 'jː', 'd̪z̪',
            't̪ː', 'mː', 'mʲː', 'rʲː', 'tʂ', 'tʲː', 'fʲː', 'fː', 'ʂː', 'dzʲː', 'd̪z̪ː', 'ʐː', 'z̪ː', 'tʂː', 'ɕ', 'ʑː',
            'ɡː', 'xː', 'zʲː', 'tsʲ']

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters_en)
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters_rus)
# symbols = [_pad] + list(_special) + list(_punctuation) + _arpabet

