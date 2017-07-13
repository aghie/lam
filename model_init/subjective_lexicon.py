"""
Functions to process a subjectivity lexicon in the format:

word\tprior_prob_v0\tprior_prob_v1\t......\tprior_prob_v2

where word is the subjective term and prior_prob_vX indicates the prior probability
of being associated to the viewpoint X

"""
import codecs
import operator
import os
import itertools

def mpqa_priors_to_dict(filepath_mpqa_priors):
    with codecs.open(filepath_mpqa_priors, encoding="utf-8") as f:
        return {l.split('\t')[0]: [float(prior) for prior in l.split('\t')[1:]] 
             for l in f.readlines()}



def mpqa_to_priors(filepath_mpqa, 
                     filepath_dest, prior_polarities = ['neutral',
                                                        'positive',
                                                        'negative',],
                     prior_prob_for_selected = 0.9):
    
    with codecs.open(filepath_dest,"w",encoding="utf-8") as f_out:
        with codecs.open(filepath_mpqa,encoding="utf-8") as f_in:
            lines = f_in.readlines()
            for l in lines:
                ls = l.strip('\n').split(' ')
                word = ls[2].split('=')[1]
                prior_polarity = ls[5].split('=')[1]
                try:                
                    index_selected_prior = prior_polarities.index(prior_polarity)
                    rest_of_priors = round((1.- prior_prob_for_selected) / (len(prior_polarities)-1),3)
                    priors = [str(prior_prob_for_selected) if i == index_selected_prior
                              else str(rest_of_priors) for i in range(0, len(prior_polarities))]
                    f_out.write('\t'.join([word,'\t'.join(priors)])+"\n")
                except ValueError:
                    pass


"""
branches per group: A list indicating how many groups to create from each prior. Alternative 
we can indicate the string 'maximum' only to split the biggest prior into to group
"""

def mpqa_synthetic_expanding(file_mpqa_priors,
                             file_mpqa_priors_synthetically_expanded,
                             branches_per_group=[2,2]):

    with codecs.open(file_mpqa_priors,encoding="utf-8") as f_in:
        with codecs.open(file_mpqa_priors_synthetically_expanded,"w",encoding="utf-8") as f_out:
            lines = f_in.readlines()
            for l in lines:
                ls = l.split('\t')
                word, priors = ls[0], [float(p) for p in ls[1:]]
                new_line = [word]
                if type(branches_per_group) == type([]):
                   # print (zip(priors,branches_per_group))
                    for p, b in zip(priors,branches_per_group):
                        for i in range(1,b+1): 
                            new_line.append(str(p / b))

                elif branches_per_group == 'maximum':
                    max_index, max_prior = index, value = max(enumerate(priors), key=operator.itemgetter(1))
                    for p, i in zip(priors,range(0,len(priors))):
                        if i == max_index:
                            new_line.append(str(p / 2))
                            new_line.append(str(p / 2))
                        else:
                            new_line.append(str(p))
                f_out.write('\t'.join(new_line)+"\n")
                


                

                
def generate_mpqa_arguing_phrases(path_dir_arguing,
                                  path_dest_phrases):
    
    macros_files = ["intensifiers.tff","modals.tff","pronoun.tff","spoken.tff","wordclasses.tff"]
    
    arguing_files = [(path_dir_arguing+os.sep+f,f) 
                     for f in os.listdir(path_dir_arguing) if
                     f.endswith('.tff') and f not in macros_files]
    
    dict_macros = {}
    
    for macro_file in macros_files:
        with codecs.open(path_dir_arguing+os.sep+macro_file) as f_macro:
            lines = f_macro.readlines()
            for l in lines[1:]:
         #       print (l)
                ls = l.split("=")
                key, values = ls[0],  [t.strip(' ') for t in ls[1].replace('\\','').replace("{","").replace("}","").strip('\r\n').strip('\r\n').split(",")]
                dict_macros[key] = values
                
                
    with codecs.open(path_dest_phrases,"w") as f_out:
        for path_arguing_file,name_arguing in arguing_files:
            with codecs.open(path_arguing_file) as f_in_arguing:
                lines = f_in_arguing.readlines()
                
                #type_file = lines[0][lines[0].find("\"")+1:lines[0].find("\"")]
                for l in lines[1:]:
                    
                    macros_in_line_values = []
                    macros_in_line_keys = []
                    for key in dict_macros:
                        if key in l:
                            if key not in macros_in_line_keys:
                                macros_in_line_values.append('|'.join(dict_macros[key]))
                                macros_in_line_keys.append(key)
                                                  
                    if macros_in_line_keys == []:
                        f_out.write('\t'.join([name_arguing.replace(".tff",""),l]))
                    else:
                        aux = zip(macros_in_line_keys,macros_in_line_values)
                        line_aux = l 
                        for key,value in aux:
                            line_aux = line_aux.replace(key,value)
               
                        f_out.write('\t'.join([name_arguing.replace(".tff",""),line_aux.replace("?","? ").replace("  "," ").replace(" )",")")]))
                        #write down all possible macro combinations
                    
                    
# generate_mpqa_arguing_phrases("/home/david.vilares/Escritorio/Papers/AtAston/research_summary/model/data/arglex_Somasundaran07/arglex_Somasundaran07/",
#                               "/home/david.vilares/Escritorio/Papers/AtAston/research_summary/model/data/arglex_Somasundaran07/all.txt")

#mpqa_to_priors("/home/david/Escritorio/Papers/AtAston/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff",
#               "/home/david/Escritorio/Papers/AtAston/research_summary/model/data/mpqa_subjectivity_priors-3-yulan-format.txt")

# mpqa_synthetic_expanding("/home/david.vilares/Escritorio/Papers/AtAston/research_summary/model/data/mpqa_subjectivity_priors-3.txt",
#                          "/home/david.vilares/Escritorio/Papers/AtAston/research_summary/model/data/mpqa_subjectivity_priors-5.txt",
#                          branches_per_group=[2,1,2])
