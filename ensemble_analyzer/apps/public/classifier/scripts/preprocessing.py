# -*- coding: utf-8 -*-

'''
  This module is intended for preprocessing the user input before aspect term extraction.
  Preprocessing eliminates noisy and irrelevant characters and words from the input of user.
'''

# Authors           : Afrahly Afable
#                     John Edrick Allas
# Calling Sequence  : preprocess(data)
# Date Written      : October 1, 2021
# Date Revised      : December 13, 2021
# Purpose           : To prerpcess the user input
# Data Structures   : Input Variable/s:
#                       - data            : LIST
#                     Output Variable/s:
#                       - preprocessed    : LIST

import os
import nltk
import pickle
from tqdm import tqdm
import re, string, json
from collections import Counter
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from nltk.tokenize import WhitespaceTokenizer
from symspellpy.symspellpy import SymSpell, Verbosity

cwd = os.getcwd()

TOKENIZER = WhitespaceTokenizer()
LEMMATIZER = WordNetLemmatizer()
ALPHABET = "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNZÑOPQRSTUVWXYZ"

contractions_filename = 'ensemble_analyzer/apps/public/data/resources/english_contractions.json'
CONTRACTIONS_LIST = json.loads(open(
    os.path.join(cwd, contractions_filename), 'r').read())

acronyms_filename = 'ensemble_analyzer/apps/public/data/resources/acronyms.json'
ACRONYMS_LIST = json.loads(open(
  os.path.join(cwd, acronyms_filename), 'r').read())

dictionary_filename = 'ensemble_analyzer/apps/public/data/resources/english_dictionary.json'
DICTIONARY_LIST = json.loads(open(os.path.join(cwd, dictionary_filename), 'r').read())

corpus_filename = 'ensemble_analyzer/apps/public/data/resources/corpus.pickle'
CORPUS = pickle.loads(open(os.path.join(cwd, corpus_filename), 'rb').read())

def words(text): 
  return re.findall(r'\w+', text.lower())

ADD_CORPUS = Counter(words(open(
  'ensemble_analyzer/apps/public/data/resources/corporaForSpellCorrection.txt').read()))

CORPUS = CORPUS + ADD_CORPUS

# Translate data
def translate_data(tweet):
  return GoogleTranslator(source='tl', target='en').translate(tweet)

# Noise Removal
# Lowercase
def lowercase(data):
  return " ".join(data.lower() for data in data.split())

# Removal of URLs
def remove_urls(data):
  url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
  text = re.sub(url_regex, '' ,data)
  return text

# Removal of Twitter handles
def remove_handles(data):
  handle_regex = r'@\S+'
  text = re.sub(handle_regex, '', data)
  return text

# Removal of all hashtags
def remove_hashtags(data):
  hashtag_regex = r'#[A-Za-z0-9Ññ\-\.\_]+'
  text = re.sub(hashtag_regex, '', data)
  return text

#Remove emojis
def remove_emoji(data):
  emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "]+", flags=re.UNICODE)
  text = emoji_pattern.sub(r'',data)
  return text

# Simplify Punctuations
def simplify_punctuation(data):
    text = str(data)
    text = re.sub(r'([-!?,;])\1+', r'\1', text)
    text = re.sub(r'\.{2,}', r'...', text)
    text = text.replace('“','"').replace('”','"')
    text = text.replace("’","'")
    return text

# Substitute or symbol
def substitute_or(data):
  new_token_list = []
  text = re.sub(r"""
                [;?!$]+  # Accept one or more copies of punctuation
                \ *           # plus zero or more copies of a space,
                """,
                " ",          # and replace it with a single space
                data, flags=re.VERBOSE)
  token_list = text.split()
  char = "/"
  for token in token_list:
    if char in token:
      token = token.replace(char, " or ")
    new_token_list.append(token)

  text = " ".join(new_token_list).strip(" ")

  return text

# Substitute and symbol
def substitute_and(data):
  new_token_list = []
  text = re.sub(r"""
                [;?!$]+  # Accept one or more copies of punctuation
                \ *           # plus zero or more copies of a space,
                """,
                " ",          # and replace it with a single space
                data, flags=re.VERBOSE)
  token_list = text.split()
  char = "&"
  for token in token_list:
    if char in token:
      token = token.replace(char, " and ")
    new_token_list.append(token)

  text = " ".join(new_token_list).strip(" ")

  return text

# Replace Contractions
def substitute_contraction(data):
  new_token_list = []
  token_list = data.split()
  for word_pos in range(len(token_list)):
    word = token_list[word_pos]
    first_upper = False
    if word[0].isupper():
      first_upper = True
    if word.lower() in CONTRACTIONS_LIST:
      replacement = CONTRACTIONS_LIST[word.lower()]
      if first_upper:
        replacement = replacement[0].upper()+replacement[1:]
      replacement_tokens = replacement.split()
      if len(replacement_tokens)>1:
        new_token_list.append(replacement_tokens[0].lower())
        new_token_list.append(replacement_tokens[1].lower())
      else:
        new_token_list.append(replacement_tokens[0])
    else:
        new_token_list.append(word)
  text = " ".join(new_token_list).strip(" ")
  return text

# Slang acronyms expansion
def substitute_acronyms(data):
	new_token_list = []
  
	text = re.sub(r"""
              [;?!$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
              """,
              " ",          # and replace it with a single space
              data, flags=re.VERBOSE)
	token_list = text.split()
	for word_pos in range(len(token_list)):
		word = token_list[word_pos]
		first_upper = False
		if word[0].isupper():
				first_upper = True
		if word in ACRONYMS_LIST:
				replacement = ACRONYMS_LIST[word.lower()]
				if first_upper:
						replacement = replacement[0]+replacement[1:]
				replacement_tokens = replacement.split()
				if len(replacement_tokens)>1:
					for i in range(len(replacement_tokens)): 
						new_token_list.append(replacement_tokens[i])
				else:
						new_token_list.append(replacement_tokens[0])
		else:
				new_token_list.append(word)
	text = " ".join(new_token_list).strip(" ")
	return text      

# Remove special characters
def remove_special_char(data):
      char_regex = r"[^a-zA-ZÑñ.,']"
      text = re.sub(char_regex, " ", data)
      text = re.sub(r"""
              [*;@?!#$>=<]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
              """,
              " ",          # and replace it with a single space
              data, flags=re.VERBOSE)
      text = text.replace('...', ' ')
      return text

def remove_numbers(data):
      text = ''.join([i for i in data if not i.isdigit()])
      return text

# Removal of excess whitespace whitespace and remove duplicates
def remove_excess_space(data):
  text = str(data)
  text = re.sub(r"//t",r"\t", text)
  text = re.sub(r"( )\1+",r"\1", text)
  text = re.sub(r"(\n)\1+",r"\1", text)
  text = re.sub(r"(\r)\1+",r"\1", text)
  text = re.sub(r"(\t)\1+",r"\1", text)
  return text.strip()

# Removal of all laughing expressions
def remove_laugh_exp(data):
  laugh_regex = r"a*ha+h[ha]*"
  text = re.sub(laugh_regex, " ", data)
  return text

# Tokenization
def tokenize(data):
  tokens = TOKENIZER.tokenize(data)
  return tokens

# Normalization

# Spelling correction
def spell_correction_text(tokens):
  max_edit_distance_dictionary= 3
  prefix_length = 4
  spellchecker = SymSpell(max_edit_distance_dictionary, prefix_length)
  spellchecker.create_dictionary(CORPUS)
  preserve_chars = ['.', ',', "'", '-',':']
  
  if len(tokens) < 1:
      return ""
  #Spell checker config
  max_edit_distance_lookup = 2
  suggestion_verbosity = Verbosity.TOP # TOP, CLOSEST, ALL
  #End of Spell checker config
  token_list = tokens
  for word_pos in range(len(token_list)):
      word = token_list[word_pos]
      first_char = word[0]
      
      if first_char not in ALPHABET:
        continue
      else:
        dict_words = DICTIONARY_LIST[first_char.lower()] 
        if first_char == "#":
          break
        if word is None:
          token_list[word_pos] = ""
          continue
        if (not '\n' in word and 
            word not in string.punctuation and 
            not is_numeric(word) and 
            not (word.lower() in spellchecker.words.keys()) and 
            word not in dict_words):
          suggestions = spellchecker.lookup(word.lower(), suggestion_verbosity,
                                            max_edit_distance_lookup)
          #Checks first uppercase to conserve the case.
          upperfirst = word[0].isupper()
          index = len(word)-1
          hasSpecialChar = True if word[index] in preserve_chars else False
          #Checks for correction suggestions.
          if len(suggestions) > 0:
            correction = suggestions[0].term
            replacement = correction
          #We call our _reduce_exaggerations function if no suggestion is found. Maybe there are repeated chars.
          else:
            replacement = reduce_exaggerations(word)
          #Takes the case back to the word.
          if upperfirst:
            replacement = replacement[0].upper()+replacement[1:]
          if hasSpecialChar:
            replacement = replacement + word[len(word)-1]
          word = replacement
          token_list[word_pos] = word
  return tokens

def reduce_exaggerations(text):
    correction = str(text)
    return re.sub(r'([\w])\1+', r'\1', correction)

def is_numeric(text):
  for char in text:
    if not (char in "0123456789" or char in ".,%$"):
      return False
  return True

# Remove non-english words
def remove_non_eng(tokens):
  new_token_list = []
  dict_words = []
  preserve_chars = ['.', ',', "'", '-',':']
  char = ''
  
  for word_pos in range(len(tokens)):
    token = tokens[word_pos]
    first_letter = token[0]
    hasSpecialChar = True if token[len(token)-1] in preserve_chars else False
    
    if not first_letter in ALPHABET:
      continue
    else:
      dict_words = DICTIONARY_LIST[first_letter.lower()]
      word = token
      if hasSpecialChar:
        word = token[:len(token)-1]
        if word in dict_words:
          word = word + token[len(token)-1]  
          new_token_list.append(word)
      else:
        if word in dict_words:
          new_token_list.append(word)

  return new_token_list

# Remove single letters
def remove_single_letters(tokens):
  new_token_list = []
  for i in range(len(tokens)):
    token = tokens[i]
    if len(token) > 1 or token == 'i':
      new_token_list.append(token)
  return new_token_list

# Lemmatization
def lemmatize(tokens):
  return [LEMMATIZER.lemmatize(word, pos='v') for word in tokens]

# POS Tagging
def pos_tagging(tokens):
  return nltk.pos_tag(tokens)

def preprocess_data(data):
  text = lowercase(data) 
  text = remove_urls(text) 
  text = remove_handles(text) 
  text = remove_hashtags(text)
  text = remove_emoji(text)
  text = simplify_punctuation(text)
  text = substitute_or(text)
  text = substitute_and(text)
  text = substitute_contraction(text) 
  text = substitute_acronyms(text) 
  text = remove_special_char(text) 
  text = remove_numbers(text) 
  text = remove_laugh_exp(text) 
  text = remove_excess_space(text)
  tokens = tokenize(text)
  tokens = spell_correction_text(tokens)
  tokens = remove_non_eng(tokens)
  tokens = remove_single_letters(tokens)
  
  return tokens

# preprocess
def preprocess(data):
  
  if isinstance(data, list):
    print("\nPREPROCESSING DATA ...")
    preprocessed = [' '.join(preprocess_data(tweet)) for tweet in tqdm(data)]
    return preprocessed
  elif isinstance(data, str):
    preprocessed = preprocess_data(data)
    return preprocessed
