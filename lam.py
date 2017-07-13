from model_init.subjective_lexicon import mpqa_priors_to_dict
from preprocess import *
from infer.polya import tune_hyper

import numpy as np
import sys
import codecs
import nltk
import json
import itertools
import pickle
import warnings

#np.random.seed(1337) 

"""
Class implementing the Latent Argument Model (LAM) model, 
as described in the paper "Detecting Perspectives in Political Debates"
"""

class LAM(object):
    
    RANDOM_SWITCH = "random"
    VARIABLE_SWITCH = "lex"
    VARIABLE_PLUS_TAG_SWITCH = "lex_pos"
    TAG_SWITCH = "pos"
    CPT_SWITCH = "cpt"
    #ARG_LEX_SWITCH and ARG_LEX_PLUS_TAG_SWITCH are not used in the paper and 
    #need to be debugged to work robustly
    ARG_LEX_SWITCH = "arglex"
    ARG_LEX_PLUS_TAG_SWITCH = "arglex_pos"
    
    
    BACKGROUND_WORD = 0
    TOPIC_WORD = 1
    VIEWPOINT_WORD = 2 
    
    MINIMUM_SIZE_FOR_TOP_SENTENCES = 10
    MAXIMUM_SIZE_FOR_TOP_SENTENCES = 70
    UPDATE_PRIORS_EVERY_X_EPOCHS = 40
    
    NOUN_TAG = 'n'
    ADJ_TAG = 'j'
    ADV_TAG = 'r'
    VERB_TAG  = 'v'
    FUNCTION_TAG = 'FUNCTION_TAG'
    
    LIST_TAGS = [NOUN_TAG,ADJ_TAG,VERB_TAG,ADV_TAG,FUNCTION_TAG]
    
    #METHODS TO EXTRACT TOP REPRESENTATIVE SENTENCES
    DISCRIMINATIVE = "discriminative"
    #DISCRIMINATIVE_HARD = "discriminative-hard"
    GENERATIVE = "generative"
    
    #METHODS FOR TRAINING
    RUN_WORD_AT_A_TIME = "word_at_a_time"
    RUN_TOPIC_WORDS_FIRST = "topics_first"

    
    #ASSIGNMENTS FOR TOPIC; VIEWPOINT AND BACKGROUND WORDS
    REGULAR_ASSIGNMENT = "regular"
    
    HIGH_VIEWPOINT_HIGH_TOPIC = True
    
    COSINE_SIMILARITY_THRESHOLD = 0.6 #Not used at the moment
    
    
    #If a sentence starts with any of these words it will be skipped to be computed
    #as a top sentence
    SENTENCES_TO_SKIP = ["amendment","amendments","line"]

    
    """
    @param docs:  A list of list string. Each doc is a list of strings
    @param n_topics: An integer. The number of topics 
    @param n_arguments: An integer. The number of arguments/perspectives
    @param refresh: An integer. LAM shows log information every "refresh" iterations 
    @param filepath_subjectivity_priors: Path to the subjectivity priors file
    @param switch_strategy: Strategy to classify between topic, argument or background
                            words, i.e. based on a random assignment, PoS tags, subjectivity lexicon,
                            Valid options - random|pos|lex|lex_pos|cpt|
    @param type_sentence_extraction: How to pick up top words and sentences. 
                                     Valid options - generative|discriminative
    @param running_method: Method to do the sampling Valid_options - word_at_a_time|topics_first|just_viewpoints
    @param argument_lexica: In progress, leave it to None
    @param doc2vec_model: In progress, leave it to None. Path to a doc2vec model for semantic similarity
    @param take_tag_into_account_for_x: True if the postag plays a role on selecting the type of word when sampling.
                                        False otherwise
    @param remove_ambiguous_viewpoint_sentences: True if top sentences occurring at more than one perspectives.
                                                 False otherwise. (needs to be polished)
    """
    def __init__(self, doc_id_map, 
                 docs, n_topic=20, n_arguments=2, refresh=1001,
                 filepath_subjectivity_priors = None, 
                 switch_strategy=VARIABLE_SWITCH,
                 type_sentence_extraction=DISCRIMINATIVE,
                 running_method = RUN_TOPIC_WORDS_FIRST,
             #    word_assignment = REGULAR_ASSIGNMENT,
                 high_viewpoint_high_topic = HIGH_VIEWPOINT_HIGH_TOPIC,
                 argument_lexica = None,
                 doc2vec_model = None,
                 take_tag_into_account_for_x = True,
                 remove_ambiguous_viewpoint_sentences = True):
        
        
        if filepath_subjectivity_priors is not None:
            self.subjective_dict_priors = mpqa_priors_to_dict(filepath_subjectivity_priors)
        else:
            self.subjective_dict_priors = {}
        print ("Preprocessing docs...") 
        
        self.docs = remove_outlines(preprocess(docs, self.subjective_dict_priors))
        
        self.doc_id_map = {}
        theta_index = 0
        aux_docs = []
        for doc_id in self.docs:
            self.doc_id_map[doc_id] = theta_index
            theta_index+=1
            aux_doc = []
            index = 0
            for word, postag in self.docs[doc_id]:
                aux_doc.append((index,word,postag)) 
                index+=1
            aux_docs.append(aux_doc)
        
        self.docs = aux_docs

        self.docs = [[(index,word, self._map_postag(postag)) for index, word,postag in doc] 
                     for doc in self.docs]
        
        self.type_sentence_extraction = type_sentence_extraction   
        self.running_method = running_method
        self.word_assignment = self.REGULAR_ASSIGNMENT
        self.filepath_subjectivity_priors = filepath_subjectivity_priors
        self.high_viewpoint_high_topic = high_viewpoint_high_topic
        self.take_tag_into_account_for_x = switch_strategy in [self.VARIABLE_PLUS_TAG_SWITCH,self.VARIABLE_SWITCH] # take_tag_into_account_for_x
        self.remove_ambiguous_viewpoint_sentences = remove_ambiguous_viewpoint_sentences
        
        self.argument_lexica = None if argument_lexica is None else self._read_metadata_file(argument_lexica, n_arguments,
                                                                                             doc_id_map) 
        self.doc2vec_model = None #Doc2Vec.load(PATH_TO_BIN_MODEL) #doc2vec_model
       
        #Vocabulary values and indexes
        self.vocab = set()
        for doc in self.docs:
            self.vocab = self.vocab.union(set([word for _,word,_ in doc]))
        self.vocab = {word:index 
                      for word,index in zip(self.vocab, range(0,len(self.vocab)))}
        self.index_vocab = {self.vocab[word]:word for word in self.vocab}
        self.docs_i = [[self.vocab[word] for (index,word,postag) in doc] 
                       for doc in self.docs]
        
        avg_doc_length = sum([len(doc) for doc in docs]) / float(len(docs))
        
        #Hyper-parameters  
        self.refresh = refresh 
        self.x_strategy = switch_strategy
        
        self.x_options = list(range(0,3))     
        self.topics = list(range(n_topic)) #Topic IDs (e.g. 0...9)
        self.arguments = list(range(n_arguments)) #Argument ID's (e.g 0...1)
        self.alpha = np.zeros(shape=n_topic) + (0.05*avg_doc_length) / n_topic
        self.beta = 0.01
        self.gamma = 0.3
        self.delta = (0.05*avg_doc_length)/(n_topic*n_arguments)
        self.epsilon = 0.01 #PRIORs FOR TAG - X matrix
        
        if self.alpha.any() <= 0 or self.beta <= 0 or self.gamma < 0 or self.delta < 0 or self.epsilon < 0:
            raise ValueError("alpha, beta, gamma and delta must be greater than zero")

        self.n_word = len(self.vocab) #Vocabulary size
        
        #Number of docs, topics and arguments
        self.n_doc = len(self.docs_i)
        self.n_topic = n_topic
        self.n_arguments = n_arguments
        
        #Number of words per x value
        self.n_total_words_per_type = np.zeros(len(self.x_options),
                                              dtype=np.int64)
        
        #Number of times that a topic t was assigned
        self.topicword_count = np.zeros(self.n_topic, dtype=np.int64) 
        #Number of times that an argument a was assigned
        self.argumentword_count = np.zeros((self.n_topic,self.n_arguments), dtype=np.int64)
        self.doc_n_words_per_type = np.zeros((self.n_doc,len(self.x_options)),
                                             dtype=np.int64)
        
        
        #Matrices of occurrences
        self.theta = np.zeros((self.n_doc,self.n_topic),dtype=np.int64)  
        self.psiT = np.zeros((self.n_word,self.n_topic),dtype=np.int64)
        self.psiA = np.zeros((self.n_word,self.n_topic, self.n_arguments))
        self.omega = np.zeros((self.n_topic,self.n_arguments)) 
        self.phi = np.zeros((self.n_word,len(self.x_options)), 
                               dtype=np.int64)       
        
        #Matrix for tag - x
        self.rho = np.zeros((len(self.LIST_TAGS),len(self.x_options)),
                            dtype=np.int64)
        
        
        self.doc_assignments = {i:[] for i in range(0,len(self.docs_i))}
         
            
        #Random initializations
        if self.word_assignment == self.REGULAR_ASSIGNMENT:
            self.regular_random_initialization()
        else:
            raise NotImplementedError("Option word_assignment not recognized")
            
            


    """
    Initializes the LAM model considering background, topic and perspective words
    """
    def regular_random_initialization(self):    
        
        argument_type = None
        
        for doc_id in range(0,len(self.docs)):
            for word_index,word, postag in self.docs[doc_id]:

                wi = self.vocab[word]
                if self.x_strategy == self.TAG_SWITCH or self.x_strategy == self.CPT_SWITCH:
                    #nouns
                    if postag.lower().startswith(self.NOUN_TAG):
                        x = self.TOPIC_WORD
                        #Adjectives, verbs, adverbs
                    elif (postag.lower().startswith(self.ADJ_TAG) or 
                          postag.lower().startswith(self.VERB_TAG) or  
                          postag.lower().startswith(self.ADV_TAG)):
                        x = self.VIEWPOINT_WORD
                    #Function words
                    else:
                        x = self.BACKGROUND_WORD
                        
                elif self.x_strategy == self.VARIABLE_SWITCH:
                    if word in self.subjective_dict_priors:
                        x = self.VIEWPOINT_WORD
                    else:
                        if self.subjective_dict_priors == {}:
                            x = np.random.choice(self.x_options)
                        else:
                            
                            x = np.random.choice(self.x_options)
                            
                elif self.x_strategy == self.RANDOM_SWITCH:
                    x = np.random.choice(self.x_options)
                    
                elif self.x_strategy == self.VARIABLE_PLUS_TAG_SWITCH:
                
                    if word in self.subjective_dict_priors and not postag.lower().startswith(self.NOUN_TAG):
                        x = self.VIEWPOINT_WORD
                    elif postag.lower().startswith(self.NOUN_TAG):
                        
                        x = np.random.choice([self.BACKGROUND_WORD,self.TOPIC_WORD,self.VIEWPOINT_WORD],
                                             p=[0.05,0.9,0.05])
                        
                        #Adjectives, verbs, adverbs
                    elif (postag.lower().startswith(self.ADJ_TAG) or 
                          postag.lower().startswith(self.VERB_TAG) or 
                          postag.lower().startswith(self.ADV_TAG)):
                        
                        x = np.random.choice([self.BACKGROUND_WORD,self.TOPIC_WORD,self.VIEWPOINT_WORD],
                                             p=[0.05,0.05,0.9])
                        
                    else:
                        
                        x =  np.random.choice([self.BACKGROUND_WORD,self.TOPIC_WORD,self.VIEWPOINT_WORD],
                                             p=[0.9,0.05,0.05])  
                else:
                    raise NotImplementedError


                #We choose a random t
                t = np.random.choice(self.topics)
                
                #To later sample the switch variable
                self.doc_n_words_per_type[doc_id][x]+=1
                self.n_total_words_per_type[x]+=1
                self.phi[wi][x]+=1

                self.rho[self.LIST_TAGS.index(postag)][x]+=1
                
                #TOPIC WORD
                if x == self.TOPIC_WORD:
                    self.psiT[wi][t]+=1
                    self.theta[doc_id][t]+=1
                    
                    self.topicword_count[t]+=1
                    self.doc_assignments[doc_id].append((wi,postag,t,-1,x))
                    
                #VIEWPOINT WORD
                elif x == self.VIEWPOINT_WORD:
                    self.theta[doc_id][t]+=1
                    
                    if (self.x_strategy == self.ARG_LEX_SWITCH or
                        self.x_strategy == self.ARG_LEX_PLUS_TAG_SWITCH):
                        
                        v = argument_type if argument_type is not None else np.random.choice(self.arguments)
                    else:                    
                        priors = self.subjective_dict_priors[word] if word in self.subjective_dict_priors else np.tile(1. / self.n_arguments, self.n_arguments) 
                        v = np.random.choice(self.arguments, p=priors)
                    
                    self.psiA[wi][t][v]+=1
                    self.argumentword_count[t][v]+=1
                    self.omega[t][v]+=1
                    self.doc_assignments[doc_id].append((wi,postag,t,v,x))
                    
                #BACKGROUND WORD
                else:
                    self.doc_assignments[doc_id].append((wi,postag,-1,-1,x))
                

                
                
                
    def _is_viewpoint(self,filepath_subjectivity):
        d = {}
        with codecs.open(filepath_subjectivity) as f:
            lines = f.readlines()
            for l in lines:
                ls = l.split('\t')
                word = ls[0]
                v_priors = ls[1:]
            
                d[word] = v_priors
            return d
                
            

    def _loglikelihood(self):      
        doc_logprobs = []
        for doc_id in range(0,len(self.docs_i)):
            word_logprobs = []
            word_index = 0
            for word_id in self.docs_i[doc_id]:
                
                _,_,t,a,_ = self.doc_assignments[doc_id][word_index]
                #If it is argument word
                if a != -1:
                    word_logprobs.append(np.log(self._compute_argument_probability(doc_id, word_id, t, a)))
                #If it is topic word
                if a == -1 and t != -1:
                    word_logprobs.append(np.log(self._compute_topic_probability(doc_id, word_id, t))) 

                word_index+=1
               
            doc_logprobs.append(np.sum(word_logprobs))
        return np.sum(doc_logprobs)
    
    """
    Updates the co-occurrences matrices before the sampling process, based on the
    current assignment
    """
    def discard_current_assignment(self,doc_id,word_id,postag,t,v,x):
        
        self.phi[word_id][x]-=1
        self.doc_n_words_per_type[doc_id][x]-=1
        self.n_total_words_per_type[x]-=1
        self.rho[self.LIST_TAGS.index(postag)][x]-=1
            
        if x == self.VIEWPOINT_WORD: 
            self.psiA[word_id,t,v]-=1
            self.argumentword_count[t][v]-=1
            self.omega[t][v]-=1
            self.theta[doc_id][t]-=1
                    
        if x == self.TOPIC_WORD:
            self.theta[doc_id][t]-=1
            self.psiT[word_id][t]-=1 
            self.topicword_count[t]-=1
        
        #If it is a background word, we do nothing

        """
    Updates the co-occurrences matrices after the sampling process, based on the
    new assignment
    """
    def add_new_assignment(self,doc_id,word_id,postag,new_t,new_v,new_x):
        
        self.phi[word_id][new_x]+=1
        self.doc_n_words_per_type[doc_id][new_x]+=1
        self.n_total_words_per_type[new_x]+=1
        self.rho[self.LIST_TAGS.index(postag)][new_x]+=1
        
     
        if new_x == self.VIEWPOINT_WORD:
            self.psiA[word_id,new_t,new_v]+=1
            self.argumentword_count[new_t][new_v]+=1
            self.omega[new_t][new_v]+=1
            self.theta[doc_id][new_t]+=1
                    
        if new_x == self.TOPIC_WORD: 
            self.theta[doc_id][new_t]+=1
            self.psiT[word_id][new_t]+=1 
            self.topicword_count[new_t]+=1
            
        #If it is a background word, we do nothing
            
        """
    Sampling method where co-occurrences are updated in order
    """
    def _run_word_at_a_time(self,epochs,sentences_original, sentences_cleaned,
                                     dict_sentences_to_procodocindex):

        for e in range(0,epochs):
            for doc_id in range(0,len(self.docs_i)):
                #For each word in the document, we do the sampling
                for position in range(0,len(self.docs_i[doc_id])):
                    
                    wi,postag,t,v, x = self.doc_assignments[doc_id][position] 
                    self.discard_current_assignment(doc_id,wi,postag,t,v,x)
                     
                    new_x, prob_new_x = self.sample_x_variable(doc_id,wi,x, postag)
                    new_t, new_v = -1,-1
                    
                    if new_x == self.TOPIC_WORD:      
                        new_t, prob_new_t = self.sampling_topic(doc_id,wi)       
                    if new_x == self.VIEWPOINT_WORD:
                        new_t, new_v = self.sample_viewpoint(doc_id,self.topics, wi)
                    
                    self.doc_assignments[doc_id][position] = (wi,postag,new_t, new_v, new_x)
                    self.add_new_assignment(doc_id,wi,postag,new_t,new_v,new_x)
                    
            self._run_log_and_update(e,sentences_original, sentences_cleaned,
                                     dict_sentences_to_procodocindex)


    """
    Sampling method where topic word co-ocurrences are updated in the first place
    """
    def _run_topics_first(self,epochs,sentences_original, sentences_cleaned,
                                     dict_sentences_to_procodocindex):

        for e in range(0,epochs):
            print ("Epoch ",e,"Log-likelihood" , self._loglikelihood()
                    ,"N topic words",  self.n_total_words_per_type[self.TOPIC_WORD],
                    "N viewpoint words", self.n_total_words_per_type[self.VIEWPOINT_WORD],
                    "N background words", self.n_total_words_per_type[self.BACKGROUND_WORD],
                    "N total words", np.sum(self.n_total_words_per_type, axis=None))
            for doc_id in range(0,len(self.docs_i)):
                
                topic_words = []
                background_words = []
                viewpoint_words = []
              
                #For each word in the document, we do the sampling
                for position in range(0,len(self.docs_i[doc_id])):
                    wi,postag,t,v,x = self.doc_assignments[doc_id][position] 
                    new_x, prob_new_x = self.sample_x_variable(doc_id,wi,x, postag)
                    
                    if new_x == self.TOPIC_WORD:
                        topic_words.append(position)
                    if new_x == self.VIEWPOINT_WORD:
                        viewpoint_words.append(position)
                    #Is background word
                    if new_x == self.BACKGROUND_WORD:
                        background_words.append(position)
                                         
                for position in background_words:
                    new_t, new_v = -1,-1
                    wi,postag,t,v,x = self.doc_assignments[doc_id][position] 
                    self.discard_current_assignment(doc_id,wi,postag,t,v,x)
                    self.doc_assignments[doc_id][position] = (wi,postag,new_t, new_v, self.BACKGROUND_WORD)
                    self.add_new_assignment(doc_id,wi,postag,new_t,new_v,self.BACKGROUND_WORD)
                    
                    
                doc_topics = set([])
                
                for position in topic_words:
                    new_t, new_v = -1,-1
                    wi,postag,t,v,x = self.doc_assignments[doc_id][position] 
                    self.discard_current_assignment(doc_id,wi,postag,t,v,x)
                    new_t, prob_new_t = self.sampling_topic(doc_id, wi)
                    self.doc_assignments[doc_id][position] = (wi,postag,new_t, new_v, self.TOPIC_WORD)
                    self.add_new_assignment(doc_id,wi,postag,new_t,new_v,self.TOPIC_WORD)
                    doc_topics.add(new_t)
                
                
                if doc_topics == set([]):
                    doc_topics = self.topics
                
                for position in viewpoint_words:
                    new_t, new_v = -1,-1
                    wi,postag,t,v,x = self.doc_assignments[doc_id][position] 
                    self.discard_current_assignment(doc_id,wi,postag,t,v,x)
                    new_t, new_v = self.sample_viewpoint(doc_id,doc_topics, wi)
                    self.doc_assignments[doc_id][position] = (wi,postag,new_t, new_v, self.VIEWPOINT_WORD)
                    self.add_new_assignment(doc_id,wi,postag,new_t,new_v,self.VIEWPOINT_WORD)
                
            self._run_log_and_update(e,sentences_original, sentences_cleaned,
                                     dict_sentences_to_procodocindex)
                
                
                


    def _run_log_and_update(self,e, sentences_original,
                           sentences_cleaned, dict_sentence_to_procodocindex):
        
        if e % self.refresh == 0 and e!=0:
            print ("\n\n")
            print ("Epoch... ",e,"Log-likelihood" , self._loglikelihood()
                    ,"N topic words",  self.n_total_words_per_type[self.TOPIC_WORD],
                    "N viewpoint words", self.n_total_words_per_type[self.VIEWPOINT_WORD],
                    "N background words", self.n_total_words_per_type[self.BACKGROUND_WORD],
                    "N total words", np.sum(self.n_total_words_per_type, axis=None))
                    
                    
            if self.word_assignment == self.REGULAR_ASSIGNMENT:
                self.print_top_words(path_output=None,top_t= 10)
            
        #Updating priors    
        if e % self.UPDATE_PRIORS_EVERY_X_EPOCHS == 0:

            print("Tuning (alpha)")
            #Not an elegant fix: tune_hyper might return NaN or values below 0. If so, alpha is not
            #updated
            try:
                alpha_aux = tune_hyper(self.theta, self.alpha)
                if np.isnan(alpha_aux).any(): print ("NaN found", alpha_aux)
                if len([e for e in alpha_aux if e <= 0]) == 0 and not np.isnan(alpha_aux).any():
                    self.alpha = alpha_aux
            except RuntimeWarning:
                warnings.warn("tune_hyper returned an unexpected value")




    """
    Base method to train the model
    """
    def run(self,
            sentences_original,
            sentences_cleaned, 
            dict_sentence_to_procodocindex,
            epochs=10):
        
        print ("Epoch... -1 Log-likelihood", self._loglikelihood(),
               "N topic words", self.n_total_words_per_type[self.TOPIC_WORD],
               "N viewpoint words",self.n_total_words_per_type[self.VIEWPOINT_WORD],
               "N background words", self.n_total_words_per_type[self.BACKGROUND_WORD],
               "N total words", np.sum(self.n_total_words_per_type))
        
        
        if self.running_method == self.RUN_WORD_AT_A_TIME:
            self._run_word_at_a_time(epochs,sentences_original,
                                    sentences_cleaned, dict_sentence_to_procodocindex)
        elif self.running_method == self.RUN_TOPIC_WORDS_FIRST:
            self._run_topics_first(epochs, sentences_original,
                                    sentences_cleaned, dict_sentence_to_procodocindex)
        else:
            raise NotImplementedError



    def _compute_switch_variable_probability(self,doc_id,word_id, x, postag):
        
        #1) p (background bg | document d) = the proportion of background words in document d that are currently
        #   assigned as background words   
        p_type_given_d = (self.doc_n_words_per_type[doc_id][x] + self.gamma) / (np.sum(self.doc_n_words_per_type[doc_id],axis=None) + len(self.x_options)*self.gamma)
        #2) p (background word bg | word ) = the proportion of times word bg is sampled as background word in
        #the collection 
        p_word_given_type = ( self.phi[word_id][x] + self.beta ) / ( self.n_total_words_per_type[x] + self.n_word*self.beta)
        
        #3) p ( x | tag p) = the proportion of x words that are assigned the postag p
        if self.take_tag_into_account_for_x:
            p_type_given_tag = ( (self.rho[self.LIST_TAGS.index(postag)][x] + self.epsilon)  / 
                                 (np.sum(self.rho[:][x])  + len(self.x_options)*self.epsilon) #np.sum(self.rho[:,x] not right
                               )

            
            return p_type_given_d*p_word_given_type*p_type_given_tag
        else:
            return p_type_given_d*p_word_given_type


    def sample_x_variable(self,doc_id, word_id,x,postag):

        probs = [self._compute_switch_variable_probability(doc_id,word_id,x_aux,postag)
                for x_aux in self.x_options]
        
        
        sum_probs = sum(probs)
        new_x = np.random.choice(self.x_options, p = [prob / sum_probs for prob in probs] )
        
        return (new_x, probs[new_x])


    def _compute_topic_probability(self,doc_id,word_id,topic):
        #1) p(topic t | document d) = the proportion of (topic) words in document d that are currently assigned to topic t,
        p_topic_given_d = (self.theta[doc_id][topic] + self.alpha[topic]) / (np.sum(self.doc_n_words_per_type[doc_id],axis=None) + sum(self.alpha))
        #p_topic_given_d = (self.theta[doc_id][topic] + self.alpha[topic]) / (np.sum(self.doc_n_words_per_type[doc_id] + self.alpha ,axis=None))
        #2) p(word w | topic t) = the proportion of assignments to topic t over all documents that come from this (topic) word w        
        p_word_given_t = (self.psiT[word_id][topic] + self.beta) / (self.topicword_count[topic] +  self.n_word*self.beta)
        
        return p_topic_given_d*p_word_given_t
    
    
    def sampling_topic(self, doc_id, wi):

        probs = [self._compute_topic_probability(doc_id,wi,t) for t in self.topics]
        
        #TODO: Should not enter here after solving the bug in _run_log_and_update
        if len([prob for prob in probs if prob <=0]) > 0 or np.isnan(probs).any():
            probs = [prob if (not np.isnan(prob) and prob>0) else 0 for prob in probs]
            
        sum_probs = sum(probs)
        
        #Should not happen now
        try:
            new_t = np.random.choice(self.topics, p = [prob / sum_probs for prob in probs])
        except Warning:
            warnings.warn("new_t was computed based on some invalid probabilities")
        return (new_t,probs[new_t])
                   


    def _compute_argument_probability(self, doc_id, wi, t, v):
        
        #1) p(t|d)
        p_t_given_d = (self.theta[doc_id][t] + self.alpha[t]) / (np.sum(self.doc_n_words_per_type[doc_id],axis=None) + sum(self.alpha))
        #2) p(v|t)
        p_v_given_dt = (self.omega[t][v] + self.delta) / (np.sum(self.omega[t][:]) + self.n_arguments*self.delta)
        #3) p (w | t,v)
        p_w_given_vt = ((self.psiA[wi][t][v] + self.beta) / 
                       (self.argumentword_count[t][v]+self.n_word*self.beta))

        return p_t_given_d*p_v_given_dt*p_w_given_vt

    

    def sample_viewpoint(self, doc_id, doc_topics, wi):
   
        if self.x_strategy == self.CPT_SWITCH:
            new_t = np.random.choice(list(doc_topics))
            probs = [self._compute_argument_probability(doc_id, wi, new_t, v_aux)
                     for v_aux in self.arguments]
            sum_probs = sum(probs)
            new_v = np.random.choice(self.arguments, 
                                     p = [prob / sum_probs for prob in probs])
        else:
            #We do a joint sampling of the topic and the viewpoint
            topic_viewpoints = list(itertools.product(doc_topics, self.arguments))

            probs = [self._compute_argument_probability(doc_id, wi, t_aux, v_aux)
                     for t_aux,v_aux in topic_viewpoints]
            
            if len([prob for prob in probs if prob <=0]) > 0 or np.isnan(probs).any():
                probs = [prob if (not np.isnan(prob) and prob>0) else 0 for prob in probs]
            sum_probs = sum(probs)
            #Should not happen now
            try:
                new_t, new_v = topic_viewpoints[np.random.choice(range(0,len(topic_viewpoints)), 
                                                            p = [prob / sum_probs for prob in probs])]
            except Warning:
                warnings.warn("new_t, new_v was computed with some invalid probabilities")
        return new_t, new_v



    def compute_topic_word_score(self, t, word):
        
        if self.type_sentence_extraction == self.GENERATIVE:   
            if word in self.vocab:
                word_id = self.vocab[word]
                p_w_given_t = (self.psiT[word_id][t] + self.beta) / (self.topicword_count[t] +  self.n_word*self.beta)   
            else:
                p_w_given_t = (0 + self.beta) / (self.topicword_count[t] +  self.n_word*self.beta)   
            return np.log(p_w_given_t)
                    
        elif self.type_sentence_extraction == self.DISCRIMINATIVE:# or self.type_sentence_extraction == self.DISCRIMINATIVE_HARD:
            
            topics_aux =  list(self.topics)
            topics_aux.remove(t)

            if word in self.vocab:
                word_id = self.vocab[word]
                p_w_given_t = (self.psiT[word_id][t] + self.beta) / (self.topicword_count[t] +  self.n_word*self.beta)   
            else:
                p_w_given_t = 0#(0 + self.beta) / (self.topicword_count[t] +  self.n_word*self.beta)   
                
            max_w_given_t_prime = 0
            for t_aux in topics_aux: 
                    if word in self.vocab:
                        p_w_given_t_aux = ((self.psiT[word_id][t_aux] + self.beta) / 
                                        (self.topicword_count[t_aux] +  self.n_word*self.beta))
                    
                    else:
                        p_w_given_t_aux = 0 #((0 + self.beta) / 
                                        #(self.topicword_count[t_aux] +  self.n_word*self.beta))
                    
                    max_w_given_t_prime = max(max_w_given_t_prime,
                                                p_w_given_t_aux)
                    
            w_discriminative_score = p_w_given_t / max_w_given_t_prime if (p_w_given_t != 0 and max_w_given_t_prime !=0) else 0
            
            return w_discriminative_score
        else:
            raise NotImplementedError
        
        
        
        
    def compute_argument_word_score(self,t,v,word):
        
        if self.type_sentence_extraction == self.GENERATIVE:      
            if word in self.vocab:
                word_id = self.vocab[word]
                p_w_given_vt = ((self.psiA[word_id][t][v] + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))                
            else:
                p_w_given_vt = ((0 + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))
            
            return np.log(p_w_given_vt)        
                    
        elif self.type_sentence_extraction == self.DISCRIMINATIVE:
            
            topics_aux =  list(self.topics)
            topics_aux.remove(t)
            
            arguments_aux = list(self.arguments)
            arguments_aux.remove(v)     
            
            if word in self.vocab:  
                word_id = self.vocab[word]

                p_w_given_vt = ((self.psiA[word_id][t][v] + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))                
            else:
                p_w_given_vt = 0#((0 + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))
            
            
            max_w_given_tv_primes = 0
            for t_aux in topics_aux:
                for v_aux in arguments_aux:
                    
                    if word in self.vocab:
                        word_id = self.vocab[word]
                        p_w_given_tv_aux = ((self.psiA[word_id][t_aux][v_aux] + self.beta) / 
                                                        (self.argumentword_count[t_aux][v_aux]+self.n_word*self.beta)) 
                    else:
                        p_w_given_tv_aux = 0 #((0 + self.beta) / 
                                           # (self.argumentword_count[t_aux][v_aux]+self.n_word*self.beta))
                    max_w_given_tv_primes = max(max_w_given_tv_primes,
                                                                p_w_given_tv_aux)
                    
            w_discriminative_score = p_w_given_vt / max_w_given_tv_primes if (p_w_given_vt != 0 and max_w_given_tv_primes !=0) else 0
            return w_discriminative_score
        
        
#         elif self.type_sentence_extraction == self.DISCRIMINATIVE_HARD:
# 
#             arguments_aux = list(self.arguments)
#             arguments_aux.remove(v)     
#             
#             if word in self.vocab:  
#                 word_id = self.vocab[word]
#                 p_w_given_vt = ((self.psiA[word_id][t][v] + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))                
#             else:
#                 p_w_given_vt = 0#((0 + self.beta) / (self.argumentword_count[t][v]+self.n_word*self.beta))
#             
#             
#             max_w_given_tv_primes = 0
#             for v_aux in arguments_aux:
#                 if word in self.vocab:
#                     word_id = self.vocab[word]
#                     p_w_given_tv_aux = ((self.psiA[word_id][t][v_aux] + self.beta) / 
#                                                         (self.argumentword_count[t][v_aux]+self.n_word*self.beta)) 
#                 else:
#                     p_w_given_tv_aux = 0 #((0 + self.beta) / 
#                                            # (self.argumentword_count[t_aux][v_aux]+self.n_word*self.beta))
#                 max_w_given_tv_primes = max(max_w_given_tv_primes,
#                                                                 p_w_given_tv_aux)
#                     
#             w_discriminative_score = p_w_given_vt / max_w_given_tv_primes if (p_w_given_vt != 0 and max_w_given_tv_primes !=0) else 0
#             return w_discriminative_score
        
        
        
        
        else:
            raise NotImplementedError
        
        

    def print_top_words(self, path_output=None, top_t=50, top_v=50):
        
        
        dict_topic_word_scores = {t:[] for t in self.topics}
        dict_argument_word_scores = {t:{v:[] for v in self.arguments} for t in self.topics}
        best_topic_words_to_return = {t:[] for t in self.topics}
        best_argument_words_to_return = {t:{v:[] for v in self.arguments} for t in self.topics}
        
        #Computing top topic words
        for t in self.topics: 
            for word in self.vocab:
                dict_topic_word_scores[t].append((self.compute_topic_word_score(t,word),word))
                
        for t in self.topics:    
            added_words = []
            sorted_relevant_words = sorted(dict_topic_word_scores[t], 
                                            key = lambda t : t[0],
                                            reverse=True)
            best_topic_words_to_return[t] = sorted_relevant_words[0:top_t]     
            
        #Computing top viewpoint words       
        for t in self.topics: 
            for a in self.arguments:
                for word in self.vocab:
                    dict_argument_word_scores[t][a].append((self.compute_argument_word_score(t,a,word),word))


        for t in self.topics:    
            for a in self.arguments:
                added_words = []
                sorted_relevant_words = sorted(dict_argument_word_scores[t][a], 
                                               key = lambda t : t[0],
                                               reverse=True)
                
                best_argument_words_to_return[t][a] = sorted_relevant_words[0:top_v]            

        #Printing out
        if path_output is None:
            f_out = sys.stdout
        else:
            f_out = codecs.open(path_output,"w",encoding="utf-8")
            

        for t in best_topic_words_to_return:
     
            topic_words =[]
            for score,word in best_topic_words_to_return[t]:
                topic_words.append(word)
            f_out.write("Topics "+str(t)+": "+' '.join(topic_words)+"\n")
                
            for a in best_argument_words_to_return[t]:
                argument_words = []
                for score,word in best_argument_words_to_return[t][a]:
                    argument_words.append(word)
                f_out.write("\tViewpoint "+str(a)+": "+" ".join(argument_words)+"\n")      
            f_out.write('\n')

          
          
            
    def print_doc_topic_distribution(self, path, top_words=10):
        #Document-topic distribution 
        beta = np.tile(self.beta,(self.n_doc,self.n_topic))
        doc_topic = self.theta + beta
        rows_sum = np.sum(doc_topic, axis=1)
        rows_sum_tile = np.transpose(np.tile(rows_sum, (self.n_topic,1)))
        theta = doc_topic / rows_sum_tile
        
        #Printing doc-topic distribution
        with codecs.open(path,"w") as f_doc_topic_out:
            head = "DocId"+"\t"+'\t'.join("Topic "+str(topic) for topic in range(0,self.n_topic))
            f_doc_topic_out.write(head+"\n")
            for doc_id in range(0,self.n_doc):
                out = str(doc_id)
                out += "\t"+'\t'.join(map(str,theta[doc_id]))
                f_doc_topic_out.write(out+"\n")
                
                
    def _get_theta(self):        
                           
        beta = np.tile(self.beta,(self.n_doc,self.n_topic))
        doc_topic = self.theta + beta
        rows_sum = np.sum(doc_topic, axis=1)
        rows_sum_tile = np.transpose(np.tile(rows_sum, (self.n_topic,1)))
        theta = doc_topic / rows_sum_tile 
        return theta
    
    
        def get_top_doc_topics(self,doc,top_topics = 1):
            #TODO At the moment this approach just works for top_topics = 1
            return theta.argmax(1)
        

    def compute_viewpoint_sentence_scores(self, doc_id, sentence, sentenceid, dict_tv_scores):
        """
        mode: 'generative' | 'discriminative'
        """          
        if self.type_sentence_extraction == self.GENERATIVE:         
            for t in self.topics: 
                p_t_given_d = (self.theta[doc_id][t] + self.alpha[t]) / (np.sum(self.doc_n_words_per_type[doc_id],axis=None) + sum(self.alpha))
                for v in self.arguments:
                    p_v_given_t = (self.omega[t][v] + self.delta) / (np.sum(self.omega[t][:]) + self.n_arguments*self.delta)
                    logp_sentence_given_viewpoint = 0
                        
                    #We take into account the prob of the sentence being from that topic based on the document
                    sentence_vector =None # np.zeros(self.n_word)
                    for word in sentence:
                        logp_sentence_given_viewpoint += self.compute_argument_word_score(t, v, word) # np.log(p_w_given_vt)
                    
                    logp_sentence_given_viewpoint += np.log(p_v_given_t)
                    logp_sentence_given_viewpoint += np.log(p_t_given_d) 
                    
                    
                    dict_tv_scores[t][v].append((logp_sentence_given_viewpoint / len(sentence), sentence,str(sentenceid),
                                                 sentence_vector)) 
                    

                            
        elif self.type_sentence_extraction == self.DISCRIMINATIVE: #or self.type_sentence_extraction == self.DISCRIMINATIVE_HARD:
            for t in self.topics: 
                for v in self.arguments:
                    sentence_score = 0
                    sentence_vector = None #np.zeros(self.n_word)
                    #We take into account the prob of the sentence being from that topic based on the document
                    for word in sentence:
                        sentence_score += self.compute_argument_word_score(t, v, word)
                    dict_tv_scores[t][v].append((sentence_score / len(sentence), sentence,str(sentenceid),
                                                 sentence_vector))
        
        else:
            raise NotImplementedError

    """
    It the top sentences of a collection of documents that best represents a topic-viewpoint
    """
    def print_viewpoint_top_sentence(self, sentences, 
                                     sentences_original,
                                     d_sentence_docindex,
                                     path_output,
                                     top_sentences=10
                                    ):
        
        theta = self._get_theta()
        theta_max_indexes = theta.argmax(1)
        best_viewpoint_sentences_to_return = {t:{v:[] for v in self.arguments} for t in self.topics}
        scores_viewpoint_sentences = {t:{v:[] for v in self.arguments} for t in self.topics}
        
        score_sentence_per_topic = {}
        
        for sentenceid,sentence in zip(range(0,len(sentences)),sentences):
            
            if self._skip_sentence_as_representative(sentence): continue
            
            if d_sentence_docindex[sentenceid] not in self.doc_id_map: continue 
            else:
                doc_id = self.doc_id_map[d_sentence_docindex[sentenceid]]    

            if (len(sentence) < self.MINIMUM_SIZE_FOR_TOP_SENTENCES or
            len(sentence) > self.MAXIMUM_SIZE_FOR_TOP_SENTENCES): continue
            
            self.compute_viewpoint_sentence_scores(doc_id,sentence, sentenceid, scores_viewpoint_sentences)
        
            sentence_topic_scores = {t:[] for t in self.topics}
            self.compute_topic_sentence_scores(doc_id, sentence, sentenceid, sentence_topic_scores)
        
            score_sentence_per_topic[sentenceid] = [sentence_topic_scores[t][0][0]
                                                    for t in self.topics]

        #Getting top sentences
        for t in self.topics:    
            for v in self.arguments:
                added_sentences = []
                sorted_relevant_sentences = sorted(scores_viewpoint_sentences[t][v], 
                                                 key = lambda t:t[0],
                                                 reverse=True)
                
                if self.doc2vec_model is None:
                    if self.high_viewpoint_high_topic:
                        for score,sentence,sentence_id, vector in sorted_relevant_sentences:
                            if score_sentence_per_topic[int(sentence_id)][t] == max(score_sentence_per_topic[int(sentence_id)]):
                                best_viewpoint_sentences_to_return[t][v].append((score,sentence,sentence_id, vector))     
                    else:
                        best_viewpoint_sentences_to_return[t][v] = sorted_relevant_sentences[0:top_sentences] 
                else:
                    #We pick up the top sentences that are not to close from a doc2vec point of view
                    i = 0 
                    picked_sentences = []
                    
                    while len(picked_sentences) < top_sentences and i < len(sorted_relevant_sentences):

                        selected_sentence = sorted_relevant_sentences[i][1]
                        too_similar = False

                        selected_vector = self.doc2vec_model.infer_vector(selected_sentence).reshape(1,-1)
                        i+=1
                        
                        for _,picked_sentence,_,_ in picked_sentences:
                            picked_vector = self.doc2vec_model.infer_vector(picked_sentence).reshape(1,-1)
                            similarity = cosine_similarity(picked_vector, selected_vector).reshape(-1,1)

                            if similarity > self.COSINE_SIMILARITY_THRESHOLD:
                                too_similar = True
                                break
                            
                        if not too_similar: picked_sentences.append(sorted_relevant_sentences[i])
                          
                    best_viewpoint_sentences_to_return[t][v] = picked_sentences


        #Removing sentences that occur in both viewpoints, in the first top_sentences
        if self.remove_ambiguous_viewpoint_sentences:
            
            best_non_ambiguous = {}
            for t in self.topics:
                best_non_ambiguous[t] = {}
                for v in self.arguments:
                    best_non_ambiguous[t][v] = []
                    
                    for score,sentence,sentenceid,vector in best_viewpoint_sentences_to_return[t][v]:
                        aux_arguments = list(self.arguments)
                        aux_arguments.remove(v)
                        ambiguous = False
                        for v_aux in aux_arguments:
                            if sentence in [s for (_,s,_,_) in best_viewpoint_sentences_to_return[t][v_aux][0:top_sentences]]:
                                ambiguous = True
                                break
                        if not ambiguous:
                            best_non_ambiguous[t][v].append((score,sentence,sentenceid,vector))

                        if len(best_non_ambiguous[t][v]) > top_sentences:
                            break
        
            best_viewpoint_sentences_to_return = best_non_ambiguous                    
                                  
        

        if path_output is None:
            f_out = sys.stdout
        else:
            f_out = codecs.open(path_output,"w",encoding="utf-8")
        for topic in best_viewpoint_sentences_to_return:
            for viewpoint in best_viewpoint_sentences_to_return[topic]:
                f_out.write("\n")
                f_out.write("Topic: "+str(topic)+" Viewpoint: "+str(viewpoint)+"\n")
                f_out.write("\n")
                for prob,processed_text,sentenceid, sentence_vector  in best_viewpoint_sentences_to_return[topic][viewpoint][0:top_sentences]:
                    f_out.write(d_sentence_docindex[int(sentenceid)]+"\t"+sentences_original[int(sentenceid)]+"\t("+str(prob)+")\n")                    



    def compute_topic_sentence_scores(self, doc_id, sentence, sentenceid, dict_t_scores):
        """
        mode: 'generative' | 'discriminative'
        """          
        
        if self.type_sentence_extraction == self.GENERATIVE:         
        
            for t in self.topics: 
                p_t_given_d = (self.theta[doc_id][t] + self.alpha[t]) / (np.sum(self.doc_n_words_per_type[doc_id],axis=None) + sum(self.alpha))
                logp_sentence_given_topic = 0
                sentence_vector =None # np.zeros(self.n_word)
                for word in sentence:
                    logp_sentence_given_topic += self.compute_topic_word_score(t, word) #np.log(p_w_given_t)

                #If there is not any in-vocabulary word, we skip that sentence
                logp_sentence_given_topic += np.log(p_t_given_d) 
                    
                dict_t_scores[t].append((logp_sentence_given_topic / len(sentence), sentence,str(sentenceid), sentence_vector))
                    
        elif self.type_sentence_extraction == self.DISCRIMINATIVE:# or self.type_sentence_extraction == self.DISCRIMINATIVE_HARD:
            for t in self.topics: 
                sentence_score = 0  
                sentence_vector = None #np.zeros(self.n_word)
                #We take into account the prob of the sentence being from that topic based on the document
                for word in sentence:
                    sentence_score += self.compute_topic_word_score(t, word)

                dict_t_scores[t].append((sentence_score / len(sentence), sentence,str(sentenceid), sentence_vector))
        else:
            raise NotImplementedError        


            
    
    def _skip_sentence_as_representative(self,sentence):

        for non_representative_string in self.SENTENCES_TO_SKIP:
            if ' '.join(sentence).startswith(non_representative_string.lower()):
                return True
        return False
        
      
    
                        
    """
    It the top sentences of a collection of documents that best represents a topic-viewpoint
    """
    def print_topic_top_sentence(self, sentences, 
                                 sentences_original,
                                 d_sentence_docindex,
                                 path_output,
                                 top_sentences=10):
        
        
        theta = self._get_theta()
        theta_max_indexes = theta.argmax(1)
        best_topic_sentences_to_return = {t:[] for t in self.topics}
        
        scores_topic_sentences = {t:[] for t in self.topics}
                   
        for sentenceid,sentence in zip(range(0,len(sentences)),sentences):
            
            if self._skip_sentence_as_representative(sentence): continue
            
            if d_sentence_docindex[sentenceid] not in self.doc_id_map: continue 
            else:
                doc_id = self.doc_id_map[d_sentence_docindex[sentenceid]]    
            representative_topic = theta_max_indexes[doc_id]
             
            if (len(sentence) < self.MINIMUM_SIZE_FOR_TOP_SENTENCES or
            len(sentence) > self.MAXIMUM_SIZE_FOR_TOP_SENTENCES): continue

            self.compute_topic_sentence_scores(doc_id,sentence, sentenceid, scores_topic_sentences)
                
        for t in self.topics:
            sorted_relevant_sentences = sorted(scores_topic_sentences[t], 
                                               key = lambda t:t[0],
                                               reverse=True)

            best_topic_sentences_to_return[t] = sorted_relevant_sentences[0:top_sentences]         

        
            
        if path_output is None:
            f_out = sys.stdout
        else:
            f_out = codecs.open(path_output,"w",encoding="utf-8")
            
        for topic in best_topic_sentences_to_return:
            f_out.write("\n")
            f_out.write("Topic: "+str(topic)+"\n")
            f_out.write("\n")
            for prob,processed_text,sentenceid, sentence_vector  in best_topic_sentences_to_return[topic]:

                f_out.write(d_sentence_docindex[int(sentenceid)]+"\t"+sentences_original[int(sentenceid)]+"\t("+str(prob)+")\n")
       
    
            
            
#    def save(self, filepath):
#        """
#        Save the various model distribtions as a json file at filepath
#        """
#        data = {
#                "vocab" : self.vocab,
#                "index_vocab": self.index_vocab,
#                "theta" : self.theta.tolist(),
#                "psiT" : self.psiT.tolist(),
#                "word_view_topic" : self.psiA.tolist(),
#                "alpha": self.alpha.tolist(),
#                "beta" : self.beta,
#                "gamma" : self.gamma,
#                "delta": self.delta,
#                "docs" : self.docs,
# #               "assigments": [(str(k), str(w,t,v)) for k,(w,t,v) in self.doc_assignments.items()]
#            }
#          
#        with open(filepath, "w") as f:
#            json.dump(data, f, indent = 4)
            
            
            
            
    def _read_metadata_file(self,path_metadata, n_arguments,
                            doc_id_map):
        
        dict_metadata = {}
        with codecs.open(path_metadata) as f_metadata:
            lines = f_metadata.readlines()
            
            #get most common argument types
            argument_occ = {}
            for l in lines:
                ls = l.strip('\n').split('\t')
                doc_id, index, argument = ls[0], ls[2], ls[3]

                if argument not in argument_occ:
                    argument_occ[argument] = 0
                argument_occ[argument]+=1
    
            most_common_types =[key for key, value in 
                                sorted(argument_occ.items(), 
                                       key= lambda t: t[1], 
                                       reverse=True)[:n_arguments]]



            for l in lines:
                ls = l.strip('\n').split('\t')
                try:
                    doc_id, index, argument = doc_id_map[ls[0]], int(ls[2]), ls[3]
                except KeyError:
                    #This represents a document that contained argument words but
                    #it was too long or too short and was removed by preprocess.py
                    pass
                if argument not in most_common_types:
                    continue
                if doc_id not in dict_metadata:
                    dict_metadata[doc_id] = {}
                dict_metadata[doc_id][index] = {}
                dict_metadata[doc_id][index] = most_common_types.index(argument)
             
        return dict_metadata



    def _map_postag(self,postag):
        if postag.lower().startswith(self.NOUN_TAG):
            return self.NOUN_TAG
        if postag.lower().startswith(self.VERB_TAG):
            return self.VERB_TAG
        if postag.lower().startswith(self.ADV_TAG):
            return self.ADV_TAG
        if postag.lower().startswith(self.ADJ_TAG):
            return self.ADJ_TAG
        return self.FUNCTION_TAG





                
