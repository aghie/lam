import string
import nltk
import sys
import numpy as np 
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize import TweetTokenizer
from stop_words import get_stop_words
from collections import OrderedDict
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import StanfordNERTagger
from model_init.subjective_lexicon import mpqa_priors_to_dict

stopwords = get_stop_words("en")

#Stopwords typically occurring in debates
stopwords += [
  "gentleman",
  "hon",
  "right",
  "rose—",
  "minister",
  "member",
  "members",
  "mr",
  "mrs",
  "friend",
  "mr_speaker",
  "mr","speaker",
  "chief_secretary",
  "her_majesty",
  "her","majesty",
  "house_of_commons",
  "house_of_lords",
]


"""
Remove documents that are either too long or too short
"""
def remove_texts(docs,high=60, low=10, strategy='zipfs'):
    lengths = [len(doc) for doc_id, doc in docs.items()]
    new_docs = OrderedDict({}) 
    
    
    mean = np.mean(lengths)
    std = np.std(lengths)
    print("------------------------------",
        "Printing some texts statistics",
        "------------------------------")

    print ("Mean:",mean,"Std: +/-",std)

    for percentile in range(100,10,-5):
      percentile_value = np.percentile(lengths, percentile)
      print ("Percentile "+str(percentile)+" = "+str(percentile_value))
    
    if strategy == 'mean-std': #not good, std desviation is too large
        for doc_id, doc in docs.items():
            if mean < len(doc) and len(doc) < mean+2*std:
                new_docs[doc_id] = doc
    elif strategy == 'zipfs':
        for doc_id, doc in docs.items():
            if low <= len(doc) and len(doc) <= high:
                new_docs[doc_id] = doc
    elif strategy == 'percentile':        
        for doc_id, doc in docs.items():
            if np.percentile(lengths,15) <= len(doc) and len(doc) <= np.percentile(lengths,95):
                new_docs[doc_id] = doc        
    
    return new_docs


"""
Removes the words that are too common or too rare
Each doc_i is a list of tuples (textid,[(word,tag)])
"""
def remove_words(docs, high = 120, low =0, strategy='mean-std'):
  
  vocab = {}
  for doc_id, doc in docs.items():
    for word,postag in doc:
      if word in vocab:
        vocab[word] += 1
      else:
        vocab[word] = 1
  lengths = np.array([vocab[word] for word in vocab])
  sorted_lengths = sorted(lengths,reverse=True)
  mean = np.mean(lengths)
  std = np.std(lengths)

  print("------------------------------",
        "Printing some word statistics",
        "------------------------------")

  print ("Mean:",mean,"Std: +/-",std)

  for percentile in range(100,10,-5):
      percentile_value = np.percentile(lengths, percentile)
      print ("Percentile "+str(percentile)+" = "+str(percentile_value))

  new_docs = OrderedDict({})
  if strategy == 'mean-std':
       for doc_id, doc in docs.items():
           new_words = [(word,postag) for word,postag in doc 
                        if (mean-2*std) < vocab[word] and vocab[word] < (mean+2*std)] 
           if new_words != []:
               new_docs[doc_id] = new_words
  elif strategy == 'zipfs':
      for doc_id, doc in docs.items():
        new_docs[doc_id] = []
        for word,postag in doc:
          if vocab[word] <= high and vocab[word] >= low:
            new_docs[doc_id].append((word,postag))
  elif strategy == 'percentile':
      for doc_id, doc in docs.items():
        new_docs[doc_id] = []
        for word,postag in doc:
          if vocab[word] <= np.percentile(lengths,99) and vocab[word] >= np.percentile(lengths,65):
            new_docs[doc_id].append((word,postag))
      
  return new_docs



"""
Determines if a string s is a number
"""
def is_number(s):
    try:
        complex(s) 
    except ValueError:
        return False

    return True


def _is_content_word(self,postag):
    return postag.startswith('n') or postag.startswith('jj') or postag.startswith('vb') or postag.startswith('rb')

"""
Lemmatizes a list of documents
Each doc_i is a list of tuples (textid,[(word,tag)])
"""
def lemmatize_words(docs):
    
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    new_docs = OrderedDict({})
    for doc_id, doc in docs.items():
        new_docs[doc_id] = []
        for word,postag in doc:
            try:
                lemma = wordnet_lemmatizer.lemmatize(word,pos=postag.lower()[0])
                new_docs[doc_id].append((lemma.lower(),postag))
            except KeyError:
                lemma = word
                new_docs[doc_id].append((lemma.lower(),postag))
                
    return new_docs

"""
Removes the stopwords from a list of documents
Each doc_i is a list of tuples (textid,[(word,tag)])
"""
def remove_stopwords(docs):
  punctuation = string.punctuation + string.digits+"”"+"“"
  new_docs = OrderedDict({})
  for doc_id, doc in docs.items():
    new_docs[doc_id] = []
    for word,postag in doc:
      if word not in stopwords and word not in punctuation and not is_number(word) and len(word) > 1: #To avoid rare web characters that might be not considered among the stopword lists
        new_docs[doc_id].append((word,postag))
  return new_docs




# """
# Returns a list o docs annotated with NER information
# Each doc_i is a list of tuples (textid,[(word,tag)])
# """
# def NER(docs):
# 
#     print ("NER... (it might take some seconds/minutes)")
#     st = StanfordNERTagger('/home/david.vilares/Descargas/stanford-ner-2012-11-11-nodistsim/conll.closed.iob2.crf.ser.gz',
#                            '/home/david.vilares/Descargas/stanford-ner-2015-12-09/stanford-ner.jar',
#                            encoding='utf-8')
#     
#     new_docs = OrderedDict({})
#     #We append all docs not to be calling ther NER jar for every single document
#     aux_docs = []
#     docs_id = []
#     for doc_id, doc in docs.items():
#         aux_docs.append(doc)
#         docs_id.append(doc_id)
#     ner_docs = st.tag_sents(aux_docs)
#     
#     if len(docs_id) != len(ner_docs): raise ValueError
#     #We can do this zip because we assumed docs is an ordered dict!
#     for doc_id,ner_doc in zip(docs_id,ner_docs):
#         composed_ner = []
#         aux = []
#         for word, ner in ner_doc:
#             if len(word) > 0:
#                 if ner == 'O':
#                     #If we finished computing a multiword NER 
#                     #we needed to put it in the list first
#                     if composed_ner != []:
#                         aux.append('_'.join(composed_ner))
#                         composed_ner = []
#                     aux.append(word)
#                 else:
#                     if ner.startswith('B-') and composed_ner != []:
#                         aux.append('_'.join(composed_ner))
#                         composed_ner = [word]
#                     else:
#                         composed_ner.append(word)
#             new_docs[doc_id] = aux
#     return new_docs
    
"""
Each doc_i is a list of tuples (textid,[(word,tag)])
""" 
def lowercase(docs):
    new_docs = OrderedDict({})
    for doc_id, doc in docs.items():
        new_docs[doc_id] = [(w.lower(),p) for w,p in doc]
    return new_docs
    

def tokenize(docs, punctuation = string.punctuation, 
             tokenizer = TweetTokenizer()):  

  new_docs = OrderedDict({})
  for doc_id, doc in docs.items():
    new_docs[doc_id] = tokenizer.tokenize(doc)
  return new_docs


"""
Tags a collection of documents
Each doc_i is a list of tuples (textid, text)
"""
def postagging(docs):
    
   new_docs = OrderedDict({})
   for doc_id, doc in docs.items():
       new_docs[doc_id] = nltk.pos_tag(docs[doc_id])
   return new_docs


def _index_inside_negating_scope(index, indexes_negating, scope = 4):
    
    aux = False
    for index_negating in indexes_negating:
        if index_negating < index and index_negating + scope > index:
            return True
    return aux

"""
Applies a simple negation heuristic to created 'negation bi-grams' (e.g. 
transforms not good into not_good)
Each doc_i is a list of tuples (word,tag)
"""
def neg_subjective(docs, negating_terms=['not','no'], subjective_dict_priors={}):
    new_docs = OrderedDict({})

    for doc_id, doc in docs.items():
        
        new_docs[doc_id] = []
        indexes_negating = []
        #identify negating terms
        index = 0
        for word,postag in doc:
            if word in negating_terms:
                indexes_negating.append(index)
            index+=1
        
        index = 0
        for word,postag in doc:
            if _index_inside_negating_scope(index, indexes_negating) and word in subjective_dict_priors: 
                new_docs[doc_id].append(("not_"+word,postag))
                #reverse subjective priors      
                if (len(subjective_dict_priors[word]) == 2 or len(subjective_dict_priors[word]) % 2 !=0 ):
                    subjective_dict_priors["not_"+word] = subjective_dict_priors[word][::-1]
                else:
                    raise NotImplementedError("List of subjectivity priors must be 2 or odd number to be able to apply an inversion of  the subjective priors")
                
            else:
                new_docs[doc_id].append((word,postag))
            index+=1

    return new_docs


def remove_outlines(docs):
    print ("Removing most common/rare words...")
    non_outlinewords_docs = remove_words(docs, strategy='percentile')
    print("Removing the largest/shortest texts...")
    non_outline_docs = remove_texts(non_outlinewords_docs, strategy='percentile')
    return non_outline_docs


"""
Each doc is a list of tuples (textid, text)
"""
def preprocess(ori_docs, subjective_dict_priors):
    
  print ("Tokenizing...")
  tokenized_docs = tokenize(ori_docs, tokenizer= TweetTokenizer())
  print ("PoS tagging...")
  postagged_docs = postagging(tokenized_docs)
 # ner_docs = NER(tokenized_docs)
  print ("Lemmatizing...")
  lematized_docs = lemmatize_words(postagged_docs)
  print ("Lowercasing...")
  lowercase_docs = lowercase(lematized_docs)
  
  if subjective_dict_priors is not None:
      print ("Applying simple negation scope")
      
      neg_subjective_docs= neg_subjective(lowercase_docs, 
                                          negating_terms = ['not'],
                                          subjective_dict_priors=subjective_dict_priors)
  else:
      neg_subjective_docs = lowercase_docs
  print ("Removing stopwords...")
  non_stopwords_docs = remove_stopwords(neg_subjective_docs)
  
  return non_stopwords_docs

