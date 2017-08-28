from lam import LAM
from preprocess import *
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from model_init.subjective_lexicon import mpqa_priors_to_dict
from collections import OrderedDict
from stop_words import get_stop_words

import sys
import os 
import codecs
import pickle
import string
import nltk
import yaml


PATH_DOCS="path_docs"
PATH_SUBJECTIVITY_PRIORS="path_subjectivity_priors"
PATH_OUTPUT="path_output"
EPOCHS="epochs"
N_TOPICS="n_topics"
N_VIEWPOINTS="n_viewpoints"
X="x"
SENTENCE_EXTRACTION="sentence_extraction"
RUNNING_METHOD="running_method"
HIGH_VIEWPOINT_HIGH_TOPIC="high_viewpoint_high_topic"
REMOVE_AMBIGOUS_VIEWPOINT_SENTENCES="remove_ambigous_viewpoint_sentences"
PATH_PALMETTO_PYSCRIPT="path_palmetto_pyscript"
PATH_PALMETTO_JAR="path_palmetto_jar"
PATH_WIKIPEDIA_DUMP="path_wikipedia_dump"
PATH_STANFORD_TOKENIZER="path_to_stanford_tokenizer"



stopwords = get_stop_words("en")
#Some stopwords typically occurring in debates
stopwords += [
  "gentleman",
  "hon",
  "hon_friend",
  "hon_lady",
  "hon_gentleman",
  "hon_member",
  "hon_members",
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
  "secretary",
  "home_secretary",
]





def get_docs(filepath):
    
    
    docs= []
    doc_id = 0
    doc_id_map = OrderedDict({})
    with open(filepath, "r") as f:
        for line in f:
            if line[0] != "#":
                line_split = line.strip('\n').split('\t')
                try:
                    if len(line_split) > 1 :
                        docs.append(line_split[4].split())
                        doc_id_map[line_split[0]] = doc_id
                        doc_id += 1
                except:
                    pass

    return doc_id_map,docs



def get_sentences(filepath, subjective_dict_priors):
    
    d_date_procdocindex,docs = get_docs(filepath)
    sentence_tokenizer = PunktSentenceTokenizer()
    extra_abbreviations = ["hon"]
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    sentences_to_analyze = OrderedDict({})
    d_sentenceid_to_procdocindex = {}

    #We take the original texts and preprocess them
    i=0
    for date in d_date_procdocindex:        
        text = ' '.join(docs[d_date_procdocindex[date]])
        
        sentences = sentence_tokenizer.tokenize(text)

        for s in sentences:
            sentences_to_analyze[i] = s
            d_sentenceid_to_procdocindex[i] = date
            i+=1

    #We can do this because it is an ordered dict
    tokenized_sentences = tokenize(sentences_to_analyze, 
                                   tokenizer= TweetTokenizer())
    postagged_sentences = postagging(tokenized_sentences)
    lematized_sentences = lemmatize_words(postagged_sentences)
    lowercase_sentences = lowercase(lematized_sentences)
    neg_subjective_sentences = neg_subjective(lowercase_sentences,
                                          subjective_dict_priors = subjective_dict_priors) 
    stopword_sentences = remove_stopwords(neg_subjective_sentences)

    if len(sentences_to_analyze) != len(d_sentenceid_to_procdocindex):
        print ("WARNING: Some sentences are not asigned a doc id")
    

    #Transform [(word,tag)] into [word]
    aux = [stopword_sentences[key] for key in stopword_sentences]
    sentences_cleaned = [[w for w,t in s] for s in aux]

    
    #return sentences_to_analyze,[stopword_sentences[key] for key in stopword_sentences], d_sentenceid_to_procdocindex                           
    return sentences_to_analyze,sentences_cleaned, d_sentenceid_to_procdocindex

def add_metadata_to_file(path_top_sentences,path_with_metadata,
                         path_votes):
    

    with codecs.open(path_with_metadata) as f_metadata:
        lines = f_metadata.readlines()
        d_metadata = {l.split('\t')[0]:(l.split('\t')[1],
                      l.split('\t')[2],l.split('\t')[3]) for l in lines}
    
    with codecs.open(path_top_sentences) as f_top:
        lines = f_top.readlines()
        with codecs.open(path_top_sentences+"+metadata","w") as f_top_metadata:        
        
            for l in lines:
                if not l.startswith("Topic") and l!='\n':
                    ls = l.split('\t')
                    id = ls[0]
                    f_top_metadata.write(l.strip('\n')+"\t"+ '\t'.join(d_metadata[id])+"\n")
                else:
                    f_top_metadata.write(l)



if __name__ == "__main__":
    
     print ("Input arguments",sys.argv)
     
     
     config = yaml.safe_load(open(sys.argv[1]))
     path_docs = config[PATH_DOCS]
     name_docs = path_docs.split('/')[-1].split(".")[0]
     topics = [int(t) for t in config[N_TOPICS]]#.split(',')]
     viewpoints = [int(v) for v in config[N_VIEWPOINTS]]#.split(',')]
     epochs = int(config[EPOCHS])
     dest_dir = config[PATH_OUTPUT]
     try:
         path_priors = config[PATH_SUBJECTIVITY_PRIORS]
     except IndexError:
         print ("WARNING: No subjectivity priors specified")
         path_priors = None     
     switch_strategy = config[X]
     type_sentence_extraction = config[SENTENCE_EXTRACTION]
     running_method = config[RUNNING_METHOD]
    # word_assignment = config[WORD_ASSIGNMENT]
     high_viewpoint_high_topic = config[HIGH_VIEWPOINT_HIGH_TOPIC]
     argument_lexica = None #Not used in the current model
     remove_ambiguous_viewpoint_sentences =  config[REMOVE_AMBIGOUS_VIEWPOINT_SENTENCES]
     
     
     new_docs = OrderedDict({})
     docs= []
     doc_id = 0
     doc_id_map = OrderedDict({})
     with open(path_docs, "r") as f:
         for line in f:
             if line[0] != "#":
                 line_split = line.strip('\n').split('\t')
                 try:
                     if len(line_split) > 1 :
                         new_docs[line_split[0]] = line_split[-1]
                         docs.append(line_split[-1])
                         doc_id_map[line_split[0]] = doc_id
                         doc_id += 1
                 except:
                     pass
                 
 
     for num_topics in topics:
         for num_views in viewpoints:
             model = LAM(doc_id_map,
                         new_docs, n_topic = num_topics,
                         n_arguments = num_views,
                         filepath_subjectivity_priors = path_priors,
                         switch_strategy = switch_strategy,
                         type_sentence_extraction= type_sentence_extraction,
                         running_method = running_method,
                         high_viewpoint_high_topic=high_viewpoint_high_topic,
                         argument_lexica = argument_lexica,
                         remove_ambiguous_viewpoint_sentences= remove_ambiguous_viewpoint_sentences)
 
    
    
             sentences_original, sentences_cleaned, dict_sentence_to_procdocindex = get_sentences(path_docs, model.subjective_dict_priors)
             
             
             model.run(epochs=epochs,
                       sentences_original=sentences_original,
                       sentences_cleaned=sentences_cleaned, 
                       dict_sentence_to_procodocindex=dict_sentence_to_procdocindex)
             
            
             name_output = '-'.join(map(str,[switch_strategy,type_sentence_extraction,running_method,
                                             high_viewpoint_high_topic,num_topics,num_views,epochs,name_docs]))
             
             path_output = dest_dir+os.sep+name_output+".out"

             
             model.print_top_words(path_output) 
 
             path_doc_topic_distribution = dest_dir+os.sep+name_output+".doc_topic_distribution"
             

             model.print_doc_topic_distribution(path_doc_topic_distribution)
             
            

             path_out_best_sentences = dest_dir+os.sep+name_output+".vp_representatite_sentences"
             

             sentences_original,sentences_cleaned, dict_sentence_to_procodocindex =get_sentences(path_docs, 
                                                                                                 model.subjective_dict_priors)
             
         
             top_sentences = model.print_viewpoint_top_sentence(sentences_cleaned,
                                                               sentences_original,
                                                               dict_sentence_to_procdocindex,
                                                               path_out_best_sentences)
             
             add_metadata_to_file(path_out_best_sentences, path_docs, None)  

             
             path_out_best_sentences = dest_dir+os.sep+name_output+".topic_representatite_sentences"                                                 
                        
         
             sentences_original,sentences_cleaned, dict_sentence_to_procodocindex =get_sentences(path_docs,
                                                                                                 model.subjective_dict_priors)
             
             
             top_topic_sentences = model.print_topic_top_sentence(sentences_cleaned,
                                                                      sentences_original,
                                                                      dict_sentence_to_procdocindex,
                                                                      path_out_best_sentences)

             add_metadata_to_file(path_out_best_sentences, path_docs, None)             


             
             #------------------------------------------------------------------------------------------------------
            
             #DO NOT SAVE THE MODELS IF LAUNCHING MANY EXPERIMENTS; IT TAKES LOT OF SPACE
             #TODO: ALSO, SAVING THE MODEL NEEDS TO BE DEBUGGED
                     
#              path_saved_model = dest_dir+os.sep+name_docs+'_'.join(["t-"+str(num_topics),"v-"+str(num_views),
#                                                                     "e-"+str(epochs)])+".model"
#              
#              model.save(path_saved_model)
#              
#              path_saved_model_pickle = dest_dir+os.sep+name_docs+'_'.join(["t-"+str(num_topics),"v-"+str(num_views),
#                                                                     "e-"+str(epochs)])+".pickle"
#              
#  
#      
#              f = codecs.open(path_saved_model_pickle,"wb")
#              pickle.dump(model, f, 2)
#              f.close()

             
             os.system("python "+config[PATH_PALMETTO_PYSCRIPT]+" "+path_output+" local "+config[PATH_PALMETTO_JAR]+" "+
                     config[PATH_WIKIPEDIA_DUMP])
     
   
