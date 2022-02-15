# # from scripts import preprocessing

# # orig = ["My students gave this as a token of appreciation as I helped them in their research project. As a person who is awkwardly praised, such people make their hearts fat. That's why I love being a professor/teacher 🖤💖 https://t.co/y0hfV2D84c"]
# # prep = preprocessing.preprocess(orig)
# # print(prep)

# from scripts import preprocessing
# from scripts.ate_utils import get_extracted_aspect

# def myformatting(t):
#   import re
#   import textwrap
#   t=re.sub('\s+',' ',t); t=re.sub('^\s+','',t); t=re.sub('\s+$','',t)
#   t=textwrap.wrap(t,width=100,initial_indent=' '*1,subsequent_indent=' '*15)
#   s=""
#   for i in (t): s=s+i+"\n"
#   s=re.sub('\s+$','',s)
#   return(s)

# # original_test_text_list = ["Hear us pls @DepEd_PH @PhCHED Listen to our grievances !!!!! For those of you who don't havechildren, they are also having a hard time like us, I hope you know that !!!!!#AcademicFreezeNOW", 
# #                           "I think online classes are hard", 
# #                           "The company's customer service is great", 
# #                           "Goosh, I think face to face learning is draining me",
# #                           "I've never thought that online class is so tiring 😭 though we're not going to school it's mentally tiring 😭😭😭"]

# original_test_text_list = ["I really admire how productive our professors are everyday",
#                           "As time goes on, I become more lazy to teach online. I WANT ULER TO TEACH FACE TO FACE CLASSES",
#                           "The government owes an apology to all those children and their parents , and should take an immediate response towards delivering textbooks to the government schools.",
#                           "Heaps of modules for Week 2🤦",
#                           "the toxic of the professor ‘to tiiii",
#                           "On a scale of 1-10, how bad are CHED's decisions",
#                           "last two pages I'm done with all the modules ahhh❤"]
# preprocessed_test_text_list = preprocessing.preprocess(original_test_text_list)
# extracted_aspect_list,opinion_aspect_dict_list,dependency_list = get_extracted_aspect(preprocessed_test_text_list)

# print("\n\n")
# i = 0
# for j in range(len(extracted_aspect_list)):
#   print(i)
#   print(f"Original:     {myformatting(str(original_test_text_list[i]))}")
#   print(f"Preprocessed: {myformatting(str(preprocessed_test_text_list[i]))}")
#   print(f"Dependency:   {myformatting(str(dependency_list[j]))}")
#   print(f"Extracted:    {myformatting(str(extracted_aspect_list[j]))}")
#   print(f"Opinion Dict: {myformatting(str(opinion_aspect_dict_list[j]))}")
#   print("\n")
#   i += 1


# # import re
# # def get_n_grams(tokens):
# #   tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', ' '.join(tokens))
# #   tokens = tokens.split()
# #   ans = []
# #   for i in range(0,5):
# #     temp=zip(*[tokens[i:] for i in range(i)])
# #     ans.extend(' '.join(ngram) for ngram in temp)
# #   return ans

# # orig_str = ["I've never thought that online class is so tiring 😭 though we're not going to school it's mentally tiring. 😭😭😭"]
# # prep_str = preprocessing.preprocess(orig_str)
# # n_grams = get_n_grams(prep_str)
# # print(n_grams)


# # import nltk
# # from nltk.corpus import wordnet
# # from nltk.stem import WordNetLemmatizer

# # ALPHABET = "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNZÑOPQRSTUVWXYZ"
# # LEMMATIZER = WordNetLemmatizer()

# # Replace Negations
# # def replace(word, pos=None):
# #   """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
# #   synonyms = []
# #   antonyms = []
  
# #   for syn in wordnet.synsets(word):
# #     for l in syn.lemmas():
# #         synonyms.append(l.name())
# #         if l.antonyms():
# #             antonyms.append(l.antonyms()[0].name())
  
# #   print(set(synonyms))
# #   print(set(antonyms))
    
# #   antonyms = set()
# #   for syn in wordnet.synsets(word, pos=pos):
# #     for lemma in syn.lemmas():
# #       for antonym in lemma.antonyms():
# #         antonyms.add(antonym.name())
# #   if len(antonyms) == 1:
# #     return antonyms.pop()
# #   else:
# #     return None

# # def replaceNegations(data):
# #   """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
# #   i, l = 0, len(data)
# #   words = []
# #   has_char = False
# #   temp_char = ''
# #   while i < l:
# #     word = data[i]
# #     temp_word = ''
# #     if word == 'not' and i+1 < l:
# #       next_word = data[i+1]
# #       for char in next_word:
# #         if char not in ALPHABET:
# #           has_char = True
# #           temp_char = char
# #         else:
# #           temp_word += char
# #       temp_word = LEMMATIZER.lemmatize(temp_word)
# #       print(temp_word)
# #       ant = replace(temp_word)
# #       if ant:
# #         if has_char:
# #           ant = ant + temp_char
# #         words.append(ant)
# #         i += 2
# #         continue
# #       else:
# #         print(ant)
# #     words.append(word)
# #     i += 1
# #   print(words)
# #   return words

# # data = "This place is not great."
# # data = data.split()
# # replaceNegations(data)

from scripts.aspect_term_extraction import AspectTermExtraction
from scripts.preprocessing import preprocess

original_test_text_list = ["YOOOOO. PLUS IF YOU REALLY WANT TO PURSUE OR FINISH YOUR ONLINE CLASS, EVEN IF YOU JUST REDUCE OUR TIME BECAUSE ALMOST NOTHING IS REDUCED, JUST LIKE FACE TO FACE."]
preprocessed_test_text_list = preprocess(original_test_text_list)
ate = AspectTermExtraction(preprocessed_test_text_list,mode="test")
extracted_aspect_list,opinion_aspect_dict_list,dependency_list = ate.get_extracted_aspects()

def myformatting(t):
  import re
  import textwrap
  t=re.sub('\s+',' ',t); t=re.sub('^\s+','',t); t=re.sub('\s+$','',t)
  t=textwrap.wrap(t,width=100,initial_indent=' '*1,subsequent_indent=' '*15)
  s=""
  for i in (t): s=s+i+"\n"
  s=re.sub('\s+$','',s)
  return(s)

print("\n\n")
i = 0
for j in range(len(extracted_aspect_list)):
  print(i)
  print(f"Original:     {myformatting(str(original_test_text_list[i]))}")
  print(f"Preprocessed: {myformatting(str(preprocessed_test_text_list[i]))}")
  print(f"Dependency:   {myformatting(str(dependency_list[j]))}")
  print(f"Extracted:    {myformatting(str(extracted_aspect_list[j]))}")
  print(f"Opinion Dict: {myformatting(str(opinion_aspect_dict_list[j]))}")
  print("\n")
  i += 1


