# -*- coding: utf-8 -*-
from urllib.parse import urlencode
from collections import OrderedDict
import urllib.request
import numpy as np
import subprocess
import sys
import codecs
import os
import tempfile
import subprocess
import uuid

N_TOPICS = 20
MAX_WORDS = 10
N_VIEWPOINTS = 2
METRICS = ["ca","cp","cv","npmi","uci","umass"]
"""
python palmetto.py path_output local path_palmetto_jar path_palmetto_wikipedia_dump
"""

def _web_coherence_(topics, coherence_metric="cv"):

  url = "http://palmetto.aksw.org/palmetto-webapp/service/"+coherence_metric+"?words="
  results = {}

  i = 0
  if len(topics) >= MAX_WORDS:
    topics = topics[0:MAX_WORDS]
  print (url+"%20".join([t if t != "£" else "%C2%A3" for t in topics]))
  coherence = urllib.request.urlopen(url+"%20".join([t if t != "£" else "%C2%A3" for t in topics]), timeout = 160)
  coherence = coherence.read()
  print (coherence)
  results["words"] = t
  results["coherence"] = float(coherence)   
  i += 1   
  return results
  
  

def _run_local_palmetto(list_lines, dict_metrics):
    """
    TODO: Not finished
    """
    
    results = []
    
    for coherence_metric in dict_metrics:
      #  if coherence_metric != "C_V": continue

     #   tmpfile = tempfile.NamedTemporaryFile("w", encoding="utf-8")
        uuid_str = str(uuid.uuid4())
        tmpfile = codecs.open("/tmp/content_"+uuid_str+".txt","w" ,encoding="utf-8") #
        tmpresultsfile = codecs.open("/tmp/results"+uuid_str+".txt","w")
     #   tmpresultsfile = tempfile.NamedTemporaryFile("w",encoding="utf-8") #codecs.open("/tmp/results.txt")
        tmpresultsfile.write('')
       # topics_words_str = [" ".join(topic_words).strip(' ') for topic_words in topics]
        tmpfile.write("\n".join(list_lines))
      
        command = "java -Xmx2000m -jar "+palmetto_jar+"  "+wikipediadb+" "+coherence_metric+" "+tmpfile.name+" > "+tmpresultsfile.name #/tmp/results.txt" 
   #     print (command)
        tmpfile.close()
        tmpresultsfile.close()
        os.system(command)
        
        with codecs.open(tmpresultsfile.name,encoding="utf-8") as f_results:
            lines = f_results.readlines()[1:]
            for l in lines:
        #        print ("l",l)
                ls = l.split('\t')
                _, metric, words = int(ls[0].strip(' ')),float(ls[1].replace(',','.')),ls[2].strip("[").strip('\n').strip("]").replace(',',' ')
#                 if topicid not in results: results[topicid] = {}
#                 if coherence_metric not in results[topicid]: results[topicid][coherence_metric] = {}
                results.append(metric)
         #       print (results)
        os.remove(tmpfile.name)
        os.remove(tmpresultsfile.name)
    return results
  



def _local_coherence_(topics, topic_viewpoint_words, wikipedia=None, jar=None,
                      ):
    """
    TODO: Not finished
    """
    dict_metrics = OrderedDict({"C_V":"cv",
                    "C_A":"ca",
                    "C_P":"cp",
                    "NPMI":"npmi",
                    "UCI":"uci",
                    "UMass":"umass"})

 #   f_out = codecs.open(path_output,"w",encoding="utf-8")

#     dict_metrics = OrderedDict({"C_V":"cv",
#                                 "C_A":"ca"})

    topic_metric_values = {}
    metric_values  = {m:[] for m in METRICS}
    str_to_file=""

    list_words_str = []
    
    for i,t in zip(range(0,len(topics)),topics):
        list_words_str.append(" ".join(t))
        
        for viewpoint in topic_viewpoint_words[i]:
            list_words_str.append(" ".join(viewpoint))
    
    results = _run_local_palmetto(list_words_str,
                        dict_metrics)
    

    str_to_file  = ""
    topicid = 0
    metric_avg_values = OrderedDict({})
    n_viewpoints = len(topic_viewpoint_words[0])
    for line_index,line in zip(range(0,len(list_words_str)), list_words_str):
        
        metric_results  = []
        for i,metric in zip(range(len(dict_metrics)),dict_metrics):
       #     print ("picking", i, line_index,line_index+i*(len(list_words_str)),results[line_index+i*(len(list_words_str))])
            metric_results.append((dict_metrics[metric],results[line_index+i*(len(list_words_str))]))

        str_metric_results = [mname+"="+str(mvalue) for mname,mvalue in metric_results]
        
        if line_index % (n_viewpoints+1) == 0:
            str_to_file += "Topic "+str(topicid)+": "+line+"\t"+"\t".join(str_metric_results)+"\n"
            topicid+=1
            
            for mname,mvalue in metric_results:
        #        print (mname, mvalue)
                if "Topics average: " not in metric_avg_values:
                    metric_avg_values["Topics average: "] = {}
                if mname not in metric_avg_values["Topics average: "]:
                    metric_avg_values["Topics average: "][mname] = 0 
                metric_avg_values["Topics average: "][mname]+= float(mvalue) / len(topics)
                
        else:
            viewpointid = (line_index % (n_viewpoints+1)) -1
            str_to_file += "\tViewpoint "+str(viewpointid)+": "+line+"\t"+"\t".join(str_metric_results)+"\n"
            for mname,mvalue in metric_results:
                if "Viewpoint "+str(viewpointid)+" average: " not in metric_avg_values:
                    metric_avg_values["Viewpoint "+str(viewpointid)+" average: "] = {}
                if mname not in metric_avg_values["Viewpoint "+str(viewpointid)+" average: "]:
                    metric_avg_values["Viewpoint "+str(viewpointid)+" average: "][mname] = 0 
                metric_avg_values["Viewpoint "+str(viewpointid)+" average: "][mname]+= float(mvalue) / len(topics)
    
    
    #PRINTING AVERAGE       
    str_to_file+="\n"
    for average in metric_avg_values:
        metrics_aux = []
        for metric in metric_avg_values[average]:
     #       print (metric_values)
            metrics_aux.append(metric+"="+str(metric_avg_values[average][metric]))
        str_to_file+= average+'\t'.join(metrics_aux)+"\n\n"
    
    #print (str_to_file)
    
    
    return str_to_file
    

        
if __name__ == "__main__":
  
  d_topic_viewpoints = {}
  #We read the topics
  with codecs.open(sys.argv[1],encoding='utf-8') as f_in:
      topic_info = f_in.read().split('\n\n')
      n_topics = len(topic_info)-1
#       print (len(topic_info)-1)
#       exit()
      ti=0
      for t in topic_info:
        if t != '':
            topic_viewpoint_info = t.split('\n')
            n_viewpoints = len(topic_viewpoint_info)-1    
            
            topic_words = topic_viewpoint_info[0].split(':')[1].strip(' ').split(' ')    
            d_topic_viewpoints[ti] = []
            d_topic_viewpoints[ti].append(topic_words)
            for v in range(1,n_viewpoints+1):
                try:
                    viewpoint_words = topic_viewpoint_info[v].split(':')[1].strip(' ').split(' ')
                    d_topic_viewpoints[ti].append(viewpoint_words)
                except IndexError:
                    print ("WARNING: Viewpoint top words for viewpoint "+str(v)+" not found")
            ti+=1

  with codecs.open(sys.argv[1]+"_palmetto","w",encoding='utf-8') as f_out:   
      
    if len(sys.argv) >=3:

        palmetto_jar = None
        wikipediadb = None
        if len(sys.argv) == 5:
            print (sys.argv)
            palmetto_jar =  sys.argv[3]
            wikipediadb = sys.argv[4] 

        if sys.argv[2] == "local": 
            topics_words = []
            topic_viewpoint_words = {}
            for ti in d_topic_viewpoints:     
                topics_words.append(d_topic_viewpoints[ti][0][0:10])
            #    print (d_topic_viewpoints[ti], len(d_topic_viewpoints[ti]))
                if ti not in topic_viewpoint_words:
                    topic_viewpoint_words[ti] = []
                    
                for i in range(1,n_viewpoints+1):
                    topic_viewpoint_words[ti].append(d_topic_viewpoints[ti][i][0:10])
                
            f_out.write(_local_coherence_(topics_words, topic_viewpoint_words,wikipediadb,palmetto_jar))
        else:
            raise ValueError("Option for running palmetto is not valid")

    else:
          print ("ENTRA WEB")
          avg_topic_coherence = {m:0. for m in METRICS}
          for ti in d_topic_viewpoints:     
              f_out.write("Topic "+str(ti)+": "+' '.join(d_topic_viewpoints[ti][0])+"\t")
              metric_results = []
              for metric in METRICS:
                  results = _web_coherence_(d_topic_viewpoints[ti][0],coherence_metric=metric)
                  metric_results.append(round(results["coherence"],3))
                  avg_topic_coherence[metric]+= metric_results[-1] / n_topics
              f_out.write('\t'.join([m+"="+str(r) for m,r in zip(METRICS,metric_results)])+"\n")
              f_out.flush()
              
              #For viewpoint in viewpoints
              avg_viewpoint_coherence = {v:{} for v in range(n_viewpoints)}
              for v in range(1, n_viewpoints+1):
                  metric_results = []
                  avg_viewpoint_coherence[v-1] = {m:0. for m in METRICS}
                  for metric in METRICS:
                      results = _web_coherence_(d_topic_viewpoints[ti][v],coherence_metric=metric)
                      metric_results.append(round(results["coherence"],3))
                      print (metric, results["coherence"])
                      avg_viewpoint_coherence[v-1][metric]+= metric_results[-1] / n_viewpoints
                  print (METRICS,metric_results)
                  f_out.write('\tViewpoint '+str(v)+": "+' '.join(d_topic_viewpoints[ti][v])
                              +'\t'.join([m+"="+str(r) for m,r in zip(METRICS,metric_results)])+"\n")
              f_out.flush()
          
          #PRINTING TOPIC AVERAGE STATISTICS
          f_out.write('\n')
          f_out.write('Topics average: ')
          avg_scores = []
          for metric in METRICS:
              avg_scores.append(round(avg_topic_coherence[metric],3))
          f_out.write('\t'.join([m+"="+str(r) for m,r in zip(METRICS,map(str,avg_scores))] )+'\n\n')
          
          #PRINTING VIEWPOINT AVERAGE STATISTICS
          for v in range(n_viewpoints):
              f_out.write("Viewpoint"+str(v)+"average: ")
              avg_scores = []
              for metric in METRICS:
                  avg_scores.append(round(avg_viewpoint_coherence[v][metric],3))
              f_out.write('\t'.join([m+"="+str(r) for m,r in zip(METRICS,map(str,avg_scores))] )+'\n\n')
         
