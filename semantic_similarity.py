import collections
from collections import defaultdict 
import re
import sys
import os
import time
import pandas as pd
import random
import scipy
import statistics
import numpy as np
from sentence_transformers import SentenceTransformer, util, SentencesDataset, losses, evaluation, readers
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch

print("Is CUDA available : " + str(torch.cuda.is_available()))
print("GPU devices : " + str(torch.cuda.device_count()))
if torch.cuda.device_count()>1:
  print("GPU current device : " + torch.cuda.current_device())
  print("GPU device name : " + torch.cuda.get_device_name(torch.cuda.current_device()))




# import spacy
# nlp = spacy.load("en_core_web_lg")
# def replace_fatcats():
#   df_replacement=pd.read_csv('replacement_words.csv',sep=',',dtype=str)
#   for i in range(df_replacement.shape[0]):
#     print(df_replacement.iloc[i,0] + " " + df_replacement.iloc[i,1]) 
#   orig_words=df_replacement["original"].to_list()  
#   replacement_words=df_replacement["replacement"].to_list()
#   orig_words=[str(nlp(word)[0].lemma_) for word in orig_words]

#   df_input=pd.read_csv('fatcats_with_index_20DEC20.csv',sep=',',dtype=str)
#   for i in range(df_input.shape[0]):#df_input.shape[0]
#     sent=df_input.iloc[i,0]
#     doc=nlp(sent)
#     output_str=''
#     is_replaced=0
#     for token in doc:
#       if token.lemma_ in orig_words:
#         replacement_word=replacement_words[orig_words.index(token.lemma_)]
#         print(replacement_word)
#         output_str=output_str+replacement_word+' '
#         is_replaced=1
#       else:
#         output_str=output_str+token.text_with_ws
#         #token.text=replacement_word
#     if is_replaced:
#       print(doc.text)   
#       print(output_str) 
#     df_input.iloc[i,0]=output_str
#   df_input.to_csv('replaced_fatcats_with_index_20DEC20.csv', sep='\t', index=False)



def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def test(u, v):
  model = SentenceTransformer('paraphrase-distilroberta-base-v1') 
  sentence1='a'

def get_similarity_scores(fatcats_input_file, test_input_file, output_file, model_name):
  start_time = time.time()
  print("Start...")
  seperator='\t' if fatcats_input_file.split('.')[-1]=='tsv' else ','
    
  df=pd.read_csv(fatcats_input_file,sep=seperator,dtype=str)  
  sentences=[]
  #df=df[0:10]
  for i in range(df.shape[0]):#df.shape[0]):
    sentences.append(df.iloc[i,0])
  #print(sentences)

  model = SentenceTransformer(model_name)  
  sentence_embeddings = model.encode(sentences)
  print(sentence_embeddings[1].shape)
  elapsed_time = time.time() - start_time
  print('Took {:.03f} seconds for ref sentence embeddings'.format(elapsed_time)) 

  seperator='\t' if test_input_file.split('.')[-1]=='tsv' else ','  
  df_input=pd.read_csv(test_input_file,sep=seperator,dtype=str)  
  sim_scores, fatcat_sentence, sim_max, sim_min, sim_max_index, sim_min_index = [], [], [], [], [], []
  # sim_scores = []
  # sim_max = []
  # sim_min = []
  # sim_max_index = []
  # sim_min_index = [] 
  #df_input=df_input[0:100]
  for i in range(df_input.shape[0]):#df_input.shape[0]):
    if (i%100==0):
      print("Example : " + str(i))   
      elapsed_time = time.time() - start_time
      print('Took {:.03f} seconds'.format(elapsed_time)) 
    query = df_input.iloc[i,0]
    query_vec = model.encode([query])[0]
    sim=[]
    for ref_idx in range(len(sentence_embeddings)):
      sim.append(cosine(query_vec,sentence_embeddings[ref_idx]))
      #print("similarity = ", sim)
    sim_scores.append(sim)
    sim_max.append(max(sim))
    sim_min.append(min(sim))
    fatcat_sentence.append(df.iloc[sim.index(max(sim)),0])
    sim_max_index.append(sim.index(max(sim)))
    sim_min_index.append(sim.index(min(sim)))
  # with open('raw_out_03.txt', 'w') as f:  
  #   f.write("%s \n" % sim_scores)
  #   f.write("%s \n" % sim_max)
  #   f.write("%s \n" % sim_min)
  #   f.write("%s \n" % sim_max_index)
  #   f.write("%s \n" % sim_min_index)
  if ('with_sim_max' in test_input_file):  
    df_input.drop(['sim_max_model1','sim_max_model2', 'sim_max_model4'], inplace=True, axis=1)
  df_input.insert(2, "sim_scores", sim_scores)
  #df_input.insert(2, "fatcat_sentence", fatcat_sentence)
  df_input.insert(3, "sim_max", sim_max)
  df_input.insert(4, "sim_min", sim_min)
  df_input.insert(5, "sim_max_index", sim_max_index)
  df_input.insert(6, "sim_min_index", sim_min_index)
  
  df_input.to_csv(output_file, sep='\t', index=False)   
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time)) 

def generate_predictions_file(fatcats_input_file, test_input_file, output_file, model_name):
  start_time = time.time()
  print("Start...")
  seperator='\t' if fatcats_input_file.split('.')[-1]=='tsv' else ','
    
  df=pd.read_csv(fatcats_input_file,sep=seperator,dtype=str)  
  sentences=[]
  #df=df[0:10]
  for i in range(df.shape[0]):#df.shape[0]):
    sentences.append(df.iloc[i,0])
  #print(sentences)

  model = SentenceTransformer(model_name)  
  sentence_embeddings = model.encode(sentences)
  print(sentence_embeddings[1].shape)
  elapsed_time = time.time() - start_time
  print('Took {:.03f} seconds for ref sentence embeddings'.format(elapsed_time)) 

  seperator='\t' if test_input_file.split('.')[-1]=='tsv' else ','  
  df_input=pd.read_csv(test_input_file,sep=seperator,dtype=str)  
  sim_scores, fatcat_sentence, sim_max, sim_min, sim_max_index, sim_min_index, ground_truth = [], [], [], [], [], [], []

  #df_input=df_input[0:100]
  for i in range(df_input.shape[0]):#df_input.shape[0]):
    if (i%100==0):
      print("Example : " + str(i))   
      elapsed_time = time.time() - start_time
      print('Took {:.03f} seconds'.format(elapsed_time)) 
    query = df_input.iloc[i,0]
    query_vec = model.encode([query])[0]
    sim=[]
    for ref_idx in range(len(sentence_embeddings)):
      sim.append(cosine(query_vec,sentence_embeddings[ref_idx]))
      #print("similarity = ", sim)
    sim_scores.append(sim)
    sim_max.append(max(sim))
    sim_min.append(min(sim))
    fatcat_sentence.append(df.iloc[sim.index(max(sim)),0])
    if(df_input.shape[1]<1):
      sample_ground_truth='none'
    else:
      sample_ground_truth=1 if df_input.iloc[i,1] == 'psif' else 0
    ground_truth.append(sample_ground_truth)
    sim_max_index.append(sim.index(max(sim)))
    sim_min_index.append(sim.index(min(sim)))
  # with open('raw_out_03.txt', 'w') as f:  
  #   f.write("%s \n" % sim_scores)
  #   f.write("%s \n" % sim_max)
  #   f.write("%s \n" % sim_min)
  #   f.write("%s \n" % sim_max_index)
  #   f.write("%s \n" % sim_min_index)
  if ('with_sim_max' in test_input_file):  
    df_input.drop(['sim_max_model1','sim_max_model2', 'sim_max_model4'], inplace=True, axis=1)
  #df_input.insert(2, "sim_scores", sim_scores)
  offset= 1 if(df_input.shape[1]<1) else 0
  df_input.insert(2-offset, "fatcat_sentence", fatcat_sentence)
  df_input.insert(3-offset, "prediction", sim_max)
  df_input.insert(4-offset, "ground_truth", ground_truth)
  df_input.insert(5-offset, "fatcat_sentence_index", sim_max_index)
  
  df_input.to_csv(output_file, sep='\t', index=False)   
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))   

def generate_test_set(output_file):
  start_time = time.time()
  print("Start...")
  df_input=pd.read_csv(output_file,sep='\t',dtype=str)  
  df_output=pd.DataFrame()
  sentences=[]
  threshold=0.5
  psif_predictions=0
  nonsif_predictions=0
  
  df_psif=df_input[df_input['psif']=='psif']
  df_nonsif=df_input[df_input['psif']=='nonsif']
  df_psif.drop('sim_scores', inplace=True, axis=1)
  df_nonsif.drop('sim_scores', inplace=True, axis=1)
  print(df_psif.shape)
  print(df_nonsif.shape)
  df_psif.to_csv('model4_test_results_replaced_fatcats_psif.tsv', sep='\t', index=False)
  df_nonsif.to_csv('model4_test_results_replaced_fatcats_nonsif.tsv', sep='\t', index=False)    

def generate_filtered_test_set():
  start_time = time.time()
  print("Start...")
  df_psif1_input=pd.read_csv('model1_test_results_psif.tsv',sep='\t',dtype=str)  
  df_psif2_input=pd.read_csv('model2_test_results_psif.tsv',sep='\t',dtype=str)
  df_psif4_input=pd.read_csv('model4_test_results_psif.tsv',sep='\t',dtype=str)

  df_nonsif1_input=pd.read_csv('model1_test_results_nonsif.tsv',sep='\t',dtype=str)  
  df_nonsif2_input=pd.read_csv('model2_test_results_nonsif.tsv',sep='\t',dtype=str)
  df_nonsif4_input=pd.read_csv('model4_test_results_nonsif.tsv',sep='\t',dtype=str)
  th1, th2, th4 = 0.324, 0.361, 0.35



  df_psif_output=pd.DataFrame(columns=['Document','psif'])
  for i in range(df_psif1_input.shape[0]):
    if not ((float(df_psif1_input.iloc[i,2])<th1) or (float(df_psif2_input.iloc[i,2])<th2) or (float(df_psif4_input.iloc[i,2])<th4) 
      or (len(df_psif1_input.iloc[i,0].split())<3) or (len(df_psif1_input.iloc[i,0])<15) ) :
      df_psif_output=df_psif_output.append({'Document':df_psif1_input.iloc[i,0], 'psif':df_psif1_input.iloc[i,1], 'sim_max_model1':df_psif1_input.iloc[i,2], 'sim_max_model2':df_psif2_input.iloc[i,2], 'sim_max_model4':df_psif4_input.iloc[i,2]},ignore_index = True)
  
  df_nonsif_output=pd.DataFrame(columns=['Document','psif'])
  for i in range(df_nonsif1_input.shape[0]):      
      if not ((len(df_nonsif1_input.iloc[i,0].split())<3) or (len(df_nonsif1_input.iloc[i,0])<15) ) :
        df_nonsif_output=df_nonsif_output.append({'Document':df_nonsif1_input.iloc[i,0], 'psif':df_nonsif1_input.iloc[i,1], 'sim_max_model1':df_nonsif1_input.iloc[i,2], 'sim_max_model2':df_nonsif2_input.iloc[i,2], 'sim_max_model4':df_nonsif4_input.iloc[i,2]},ignore_index = True)

  print(df_psif_output.shape)
  print(df_nonsif_output.shape)  

  df_psif_output.to_csv('filtered_psif_with_sim_max.tsv', sep='\t', index=False)
  df_nonsif_output.to_csv('filtered_nonsif_with_sim_max.tsv', sep='\t', index=False)  

  # df_psif_output.to_csv('replaced_fatcats_filtered_test_psif.tsv', sep='\t', index=False)
  # df_nonsif_output.to_csv('replaced_fatcats_filtered_test_nonsif.tsv', sep='\t', index=False)    
def compute_filtered_data_accuracy():
  start_time = time.time()
  print("Start...")
  df_psif1_input=pd.read_csv('filtered_psif_with_sim_max.tsv',sep='\t',dtype=str) 
  df_nonsif1_input=pd.read_csv('filtered_nonsif_with_sim_max.tsv',sep='\t',dtype=str) 
  
  random_seed=random.randint(0, 6000)
  print("Random seed : " + str(random_seed))
  df_psif_test, df_psif_dev, df_psif_train=np.split(df_psif1_input.sample(frac=1, random_state=random_seed), [5000, 5000])
  df_nonsif_test, df_nonsif_dev, df_nonsif_train=np.split(df_nonsif1_input.sample(frac=1, random_state=random_seed), [5000, 5000])
  threshold=0.59
  psif_predictions1=0
  nonsif_predictions1=0
  psif_predictions2=0
  nonsif_predictions2=0
  psif_predictions4=0
  nonsif_predictions4=0      
  for i in range(df_psif_test.shape[0]):#df.shape[0]):
    score_max=float(df_psif_test.iloc[i,2])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions1=psif_predictions1+1
    score_max=float(df_psif_test.iloc[i,3])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions2=psif_predictions2+1
    score_max=float(df_psif_test.iloc[i,4])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions4=psif_predictions4+1            
  for i in range(df_nonsif_test.shape[0]):#df.shape[0]):
    score_max=float(df_nonsif_test.iloc[i,2])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions1=nonsif_predictions1+1
    score_max=float(df_nonsif_test.iloc[i,3])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions2=nonsif_predictions2+1
    score_max=float(df_nonsif_test.iloc[i,4])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions4=nonsif_predictions4+1            
  print(df_psif_test.shape)    
  print(df_psif_dev.shape)    
  print(df_psif_train.shape)    
  
        
  print(psif_predictions1/df_psif_test.shape[0])  
  print(nonsif_predictions1/df_nonsif_test.shape[0])
  print(((psif_predictions1/df_psif_test.shape[0])+(nonsif_predictions1/df_nonsif_test.shape[0]))/2)  

  print(psif_predictions2/df_psif_test.shape[0])  
  print(nonsif_predictions2/df_nonsif_test.shape[0])
  print(((psif_predictions2/df_psif_test.shape[0])+(nonsif_predictions2/df_nonsif_test.shape[0]))/2)  

  print(psif_predictions4/df_psif_test.shape[0])  
  print(nonsif_predictions4/df_nonsif_test.shape[0])
  print(((psif_predictions4/df_psif_test.shape[0])+(nonsif_predictions4/df_nonsif_test.shape[0]))/2)      

  # df_psif_output.to_csv('filtered_test_psif.tsv', sep='\t', index=False)
  # df_nonsif_output.to_csv('filtered_test_nonsif.tsv', sep='\t', index=False)  
def compute_test_set_accuracy(input_file,threshold):
  start_time = time.time()
  print("Start...")
  df_psif1_input=pd.read_csv(input_file,sep='\t',dtype=str) 
  df_nonsif1_input=pd.read_csv(input_file.replace('psif', 'nonsif'),sep='\t',dtype=str) 
 
  psif_predictions1=0
  nonsif_predictions1=0   
  for i in range(df_psif1_input.shape[0]):#df.shape[0]):
    score_max=float(df_psif1_input.iloc[i,3])
    if (score_max>threshold) and df_psif1_input.iloc[i,1]=='psif':
      psif_predictions1=psif_predictions1+1
          
  for i in range(df_nonsif1_input.shape[0]):#df.shape[0]):
    score_max=float(df_nonsif1_input.iloc[i,3])  
    if (score_max<=threshold) and df_nonsif1_input.iloc[i,1]=='nonsif':
      nonsif_predictions1=nonsif_predictions1+1
  print("Threshold : "+ str(round(threshold,2)))                
  print(psif_predictions1/df_psif1_input.shape[0])  
  print(nonsif_predictions1/df_nonsif1_input.shape[0])
  print(((psif_predictions1/df_psif1_input.shape[0])+(nonsif_predictions1/df_nonsif1_input.shape[0]))/2) 
  return  psif_predictions1/df_psif1_input.shape[0], nonsif_predictions1/df_nonsif1_input.shape[0],  ((psif_predictions1/df_psif1_input.shape[0])+(nonsif_predictions1/df_nonsif1_input.shape[0]))/2


def generate_train_dev_test_set():
  start_time = time.time()
  print("Start...")
  df_psif1_input=pd.read_csv('filtered_psif_with_sim_max.tsv',sep='\t',dtype=str) 
  df_nonsif1_input=pd.read_csv('filtered_nonsif_with_sim_max.tsv',sep='\t',dtype=str) 
  
  random_seed=random.randint(0, 6000)
  random_seed=5610
  print("Random seed : " + str(random_seed))
  df_psif_test, df_psif_dev, df_psif_train=np.split(df_psif1_input.sample(frac=1, random_state=random_seed), [500, 1000])
  df_nonsif_test, df_nonsif_dev, df_nonsif_train=np.split(df_nonsif1_input.sample(frac=1, random_state=random_seed), [500, 1000])
  threshold=0.5
  psif_predictions1=0
  nonsif_predictions1=0
  psif_predictions2=0
  nonsif_predictions2=0
  psif_predictions4=0
  nonsif_predictions4=0      
  for i in range(df_psif_test.shape[0]):#df.shape[0]):
    score_max=float(df_psif_test.iloc[i,2])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions1=psif_predictions1+1
    score_max=float(df_psif_test.iloc[i,3])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions2=psif_predictions2+1
    score_max=float(df_psif_test.iloc[i,4])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions4=psif_predictions4+1            
  for i in range(df_nonsif_test.shape[0]):#df.shape[0]):
    score_max=float(df_nonsif_test.iloc[i,2])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions1=nonsif_predictions1+1
    score_max=float(df_nonsif_test.iloc[i,3])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions2=nonsif_predictions2+1
    score_max=float(df_nonsif_test.iloc[i,4])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions4=nonsif_predictions4+1            
  print(df_psif_test.shape)    
  print(df_psif_dev.shape)    
  print(df_psif_train.shape)    
  

        
  print(psif_predictions1/df_psif_test.shape[0])  
  print(nonsif_predictions1/df_nonsif_test.shape[0])
  print(((psif_predictions1/df_psif_test.shape[0])+(nonsif_predictions1/df_nonsif_test.shape[0]))/2)  

  print(psif_predictions2/df_psif_test.shape[0])  
  print(nonsif_predictions2/df_nonsif_test.shape[0])
  print(((psif_predictions2/df_psif_test.shape[0])+(nonsif_predictions2/df_nonsif_test.shape[0]))/2)  

  print(psif_predictions4/df_psif_test.shape[0])  
  print(nonsif_predictions4/df_nonsif_test.shape[0])
  print(((psif_predictions4/df_psif_test.shape[0])+(nonsif_predictions4/df_nonsif_test.shape[0]))/2)      
  
  df_psif_test.to_csv('testset_filtered_psif_with_sim_max.tsv', sep='\t', index=False)
  df_psif_dev.to_csv('devset_filtered_psif_with_sim_max.tsv', sep='\t', index=False)
  df_psif_train.to_csv('trainset_filtered_psif_with_sim_max.tsv', sep='\t', index=False)
  df_nonsif_test.to_csv('testset_filtered_nonsif_with_sim_max.tsv', sep='\t', index=False)
  df_nonsif_dev.to_csv('devset_filtered_nonsif_with_sim_max.tsv', sep='\t', index=False)
  df_nonsif_train.to_csv('trainset_filtered_nonsif_with_sim_max.tsv', sep='\t', index=False)


def generate_training_data():
  start_time = time.time()
  print("Start...")
  df_fatcat=pd.read_csv('replaced_fatcats_with_index_20DEC20.tsv',sep='\t',dtype=str)
  print(df_fatcat.iloc[5186,0])
  df_psif1_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str) 
  print(df_psif1_input.iloc[0,0]) 
  df_psif2_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_psif4_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)

  df_nonsif1_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str)  
  df_nonsif2_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_nonsif4_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)
  # th1, th2, th4 = 0.324, 0.361, 0.35



  df_psif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat-sentence' ])
  df_nonsif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat-sentence' ])
  fatcat_psif_count=defaultdict(int)
  fatcat_nonsif_count=defaultdict(int)
  max_psif_repetetions=10
  max_nonsif_repetetions=10
  psif_th=0.65
  nonsif_th=0.2
  for i in range(df_psif1_input.shape[0]):#df_psif1_input.shape[0]
    print("Processing sentence in psif: " + str(i))
    #print(df_psif1_input.iloc[i,2])
    sim_scores1=[float(x) for x in df_psif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_psif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_psif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_min=[min(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]    
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    sim_scores_min_sorted_idx=np.argsort(np.array(sim_scores_min))
    # print(sim_scores_max[sim_scores_max_sorted_idx[0]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[-1]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[3]])
    # print(max(sim_scores_max))
    # print(min(sim_scores_max))
    #psif_th_median=(psif_th+sim_scores_min[sim_scores_min_sorted_idx[-1]])/2

    j=-1
    df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
     'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
    fatcat_psif_count[sim_scores_min_sorted_idx[j]]+=1    
    for j in range(len(sim_scores_min)-1):
      if (sim_scores_min[j]<nonsif_th):
        if (random.randint(0, 999)<4):  #0.7 % will generate 35000 sentences 
          df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':j, 'orig_sim_score':sim_scores_min[j],
           'orig_data':'psif', 'sim_score':0, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[j,0]},ignore_index = True)
          fatcat_nonsif_count[j]+=1        

      if sim_scores_min[j]>psif_th:
        if (random.randint(0, 999)<990): #1% generated 3000 sentences 33% generates 100000
          df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':j, 'orig_sim_score':sim_scores_min[j],
           'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[j,0]},ignore_index = True)
          fatcat_psif_count[j]+=1
    

    
   
  for i in range(df_nonsif1_input.shape[0]):#df_nonsif1_input.shape[0]
    print("Processing sentence in nonsif: " + str(i))
    sim_scores1=[float(x) for x in df_nonsif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_nonsif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_nonsif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_max=[max(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    for j in range(len(sim_scores_max)-1):  
      if (random.randint(0, 999)<4):#17 (1.7% results in ~64000 sentences)
      #Min nonsif sentence
        df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':j, 'orig_sim_score':sim_scores_max[j],
         'orig_data':'nonsif', 'sim_score':0, 'sentence':df_nonsif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[j,0]},ignore_index = True)
        fatcat_nonsif_count[j]+=1
  
  fatcat_psif_count_sorted = sorted(fatcat_psif_count.items(), key=lambda item: item[1])
  fatcat_nonsif_count_sorted = sorted(fatcat_nonsif_count.items(), key=lambda item: item[1])
  print(fatcat_psif_count_sorted)
  print(fatcat_nonsif_count_sorted)
  print("Total positive sentences :" + str(sum(fatcat_psif_count.values())))
  print("Total negative sentences :" + str(sum(fatcat_nonsif_count.values())))

  df_psif_output.to_csv('train_file_replaced_fatcats_v015.tsv', sep='\t', index=False, header=None)  

  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))  


def generate_training_data_v2():
  start_time = time.time()
  print("Start...")
  df_fatcat=pd.read_csv('replaced_fatcats_with_index_20DEC20.tsv',sep='\t',dtype=str)
  print(df_fatcat.iloc[5186,0])
  df_psif1_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str) 
  print(df_psif1_input.iloc[0,0]) 
  df_psif2_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_psif4_input=pd.read_csv('trainset_filtered_psif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)

  df_nonsif1_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str)  
  df_nonsif2_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_nonsif4_input=pd.read_csv('trainset_filtered_nonsif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)
  # th1, th2, th4 = 0.324, 0.361, 0.35



  df_psif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat_sentence' ])
  df_nonsif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat_sentence' ])
  fatcat_psif_count=defaultdict(int)
  fatcat_nonsif_count=defaultdict(int)
  max_psif_repetetions=10
  max_nonsif_repetetions=10
  psif_th=0.65
  nonsif_th=0.2
  for i in range(df_psif1_input.shape[0]):#df_psif1_input.shape[0]
    print("Processing sentence in psif: " + str(i))
    #print(df_psif1_input.iloc[i,2])
    sim_scores1=[float(x) for x in df_psif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_psif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_psif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_min=[min(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]    
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    sim_scores_min_sorted_idx=np.argsort(np.array(sim_scores_min))
    # print(sim_scores_max[sim_scores_max_sorted_idx[0]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[-1]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[3]])
    # print(max(sim_scores_max))
    # print(min(sim_scores_max))
    #psif_th_median=(psif_th+sim_scores_min[sim_scores_min_sorted_idx[-1]])/2

    j=-1
    sent_idx, fatcat_idx, orig_sim_score, orig_data, sim_score, sentence, fatcat=[],[],[],[],[],[],[]

    sent_idx.append(i)
    fatcat_idx.append(sim_scores_min_sorted_idx[j])
    orig_sim_score.append(sim_scores_min[sim_scores_min_sorted_idx[j]])
    orig_data.append(psif)
    sim_score.append(1)
    sentence.append(df_psif1_input.iloc[i,0])
    fatcat_sentence.append(df_fatcat.iloc[sim_scores_min_sorted_idx[j],0])
    fatcat_psif_count[sim_scores_min_sorted_idx[j]]+=1    
    for j in range(len(sim_scores_min)-1):
      if (sim_scores_min[j]<nonsif_th):
        if (random.randint(0, 999)<90): #1% generated 3000 sentences 33% generates 100000          
          sent_idx.append(i)
          fatcat_idx.append(j)
          orig_sim_score.append(sim_scores_min[j])
          orig_data.append(psif)
          sim_score.append(1)
          sentence.append(df_psif1_input.iloc[i,0])
          fatcat_sentence.append(df_fatcat.iloc[j,0])

          fatcat_nonsif_count[j]+=1        

      if sim_scores_min[j]>psif_th:
        if (random.randint(0, 999)<2): #0.7 % will generate 35000 sentences
          df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':j, 'orig_sim_score':sim_scores_min[j],
           'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[j,0]},ignore_index = True)
          fatcat_psif_count[j]+=1
    

    
   
  for i in range(df_nonsif1_input.shape[0]):#df_nonsif1_input.shape[0]
    print("Processing sentence in nonsif: " + str(i))
    sim_scores1=[float(x) for x in df_nonsif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_nonsif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_nonsif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_max=[max(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    for j in range(len(sim_scores_max)-1):  
      if (random.randint(0, 999)<2):#17 (1.7% results in ~64000 sentences)
      #Min nonsif sentence
        df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':j, 'orig_sim_score':sim_scores_max[j],
         'orig_data':'nonsif', 'sim_score':0, 'sentence':df_nonsif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[j,0]},ignore_index = True)
        fatcat_nonsif_count[j]+=1
  
  fatcat_psif_count_sorted = sorted(fatcat_psif_count.items(), key=lambda item: item[1])
  fatcat_nonsif_count_sorted = sorted(fatcat_nonsif_count.items(), key=lambda item: item[1])
  print(fatcat_psif_count_sorted)
  print(fatcat_nonsif_count_sorted)
  print("Total positive sentences :" + str(sum(fatcat_psif_count.values())))
  print("Total negative sentences :" + str(sum(fatcat_nonsif_count.values())))

  df_psif_output.to_csv('train_file_replaced_fatcats_v012.tsv', sep='\t', index=False)  

  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))    
  
 
def generate_dev_data():
  start_time = time.time()
  print("Start...")
  df_fatcat=pd.read_csv('replaced_fatcats_with_index_20DEC20.tsv',sep='\t',dtype=str)
  print(df_fatcat.iloc[5186,0])
  df_psif1_input=pd.read_csv('devset_filtered_psif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str) 
  print(df_psif1_input.iloc[0,0]) 
  df_psif2_input=pd.read_csv('devset_filtered_psif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_psif4_input=pd.read_csv('devset_filtered_psif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)

  df_nonsif1_input=pd.read_csv('devset_filtered_nonsif_with_simscores_repfatcats_model1.tsv',sep='\t',dtype=str)  
  df_nonsif2_input=pd.read_csv('devset_filtered_nonsif_with_simscores_repfatcats_model2.tsv',sep='\t',dtype=str)
  df_nonsif4_input=pd.read_csv('devset_filtered_nonsif_with_simscores_repfatcats_model4.tsv',sep='\t',dtype=str)
  # th1, th2, th4 = 0.324, 0.361, 0.35



  df_psif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat-sentence' ])
  df_nonsif_output=pd.DataFrame(columns=['sent_idx', 'fatcat_idx', 'orig_sim_score', 'orig_data', 'sim_score', 'sentence', 'fatcat-sentence' ])
  fatcat_psif_count=defaultdict(int)
  fatcat_nonsif_count=defaultdict(int)
  max_psif_repetetions=10
  max_nonsif_repetetions=10
  psif_th=0.6
  nonsif_th=0.3
  for i in range(df_psif1_input.shape[0]):#df_psif1_input.shape[0]
    #print(df_psif1_input.iloc[i,2])
    sim_scores1=[float(x) for x in df_psif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_psif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_psif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_min=[min(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]    
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    sim_scores_min_sorted_idx=np.argsort(np.array(sim_scores_min))
    # print(sim_scores_max[sim_scores_max_sorted_idx[0]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[-1]])
    # print(sim_scores_max[sim_scores_max_sorted_idx[3]])
    # print(max(sim_scores_max))
    # print(min(sim_scores_max))
    psif_th_median=(psif_th+sim_scores_min[sim_scores_min_sorted_idx[-1]])/2
    nonsif_idx=-1
    nonsif_idx_flag=0
    for j in range(len(sim_scores_min)):
      if (sim_scores_min[sim_scores_min_sorted_idx[j]]>nonsif_th) and (nonsif_idx_flag==0):
        nonsif_idx=j
        nonsif_idx_flag=1
      if sim_scores_min[sim_scores_min_sorted_idx[j]]>psif_th:
        # print(sim_scores_min_sorted_idx[j])
        # print(sim_scores_min[sim_scores_min_sorted_idx[j]])
        break
    # print(sim_scores_min_sorted_idx[j])
    # print(sim_scores_min_sorted_idx[nonsif_idx])
    # print(sim_scores_min[sim_scores_min_sorted_idx[j]])
    # print(sim_scores_min[sim_scores_min_sorted_idx[nonsif_idx]])
    if(j<len(sim_scores_min)-1):
      psif_median_idx=round((j+len(sim_scores_min)-1)/2)
      #Min psif sentence
      df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
       'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
      fatcat_psif_count[sim_scores_min_sorted_idx[j]]+=1
      #Median psif sentence
      j=psif_median_idx
      df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
       'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
      fatcat_psif_count[sim_scores_min_sorted_idx[j]]+=1
    #Max psif sentence (The highest sgould always match irrespective of threshold as thi sis psif labeled)
    j=-1
    df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
     'orig_data':'psif', 'sim_score':1, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
    fatcat_psif_count[sim_scores_min_sorted_idx[j]]+=1
    
    if(nonsif_idx>0):  
      if (random.randint(0, 10)<5):
        #Min nonsif sentence
        j=0
        df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
         'orig_data':'psif', 'sim_score':0, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
        fatcat_nonsif_count[sim_scores_min_sorted_idx[j]]+=1
      else:
        #Median nonsif sentence
        j=round(nonsif_idx/2)
        df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
         'orig_data':'psif', 'sim_score':0, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
        fatcat_nonsif_count[sim_scores_min_sorted_idx[j]]+=1
      #Median nonsif sentence
      # j=nonsif_idx
      # df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_min_sorted_idx[j], 'orig_sim_score':sim_scores_min[sim_scores_min_sorted_idx[j]],
      #  'orig_data':'psif', 'sim_score':0, 'sentence':df_psif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_min_sorted_idx[j],0]},ignore_index = True)
      # fatcat_nonsif_count[sim_scores_min_sorted_idx[j]]+=1      
    
  for i in range(df_nonsif1_input.shape[0]):#df_nonsif1_input.shape[0]
    #print(df_psif1_input.iloc[i,2])
    sim_scores1=[float(x) for x in df_nonsif1_input.iloc[i,2].strip('[]').split(',')]
    sim_scores2=[float(x) for x in df_nonsif2_input.iloc[i,2].strip('[]').split(',')]
    sim_scores4=[float(x) for x in df_nonsif4_input.iloc[i,2].strip('[]').split(',')]
    j=3
    # print(str(sim_scores1[j]) +" " +str(sim_scores2[j]) +" " +str(sim_scores4[j]) )
    sim_scores_max=[max(x) for x in zip(sim_scores1,sim_scores2,sim_scores4) ]
    # print(sim_scores_max[j])
    # print(sim_scores_min[j])
    sim_scores_max_sorted_idx=np.argsort(np.array(sim_scores_max))

    nonsif_th=0   
    if (random.randint(0, 10)<5):
    #Min nonsif sentence
      j=0
      df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_max_sorted_idx[j], 'orig_sim_score':sim_scores_max[sim_scores_max_sorted_idx[j]],
       'orig_data':'nonsif', 'sim_score':0, 'sentence':df_nonsif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_max_sorted_idx[j],0]},ignore_index = True)
      fatcat_nonsif_count[sim_scores_max_sorted_idx[j]]+=1
    else:
      #Median nonsif sentence
      j=round(len(sim_scores_max)/2)
      df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_max_sorted_idx[j], 'orig_sim_score':sim_scores_max[sim_scores_max_sorted_idx[j]],
       'orig_data':'nonsif', 'sim_score':0, 'sentence':df_nonsif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_max_sorted_idx[j],0]},ignore_index = True)
      fatcat_nonsif_count[sim_scores_max_sorted_idx[j]]+=1
    #Median nonsif sentence
    j=-1
    df_psif_output=df_psif_output.append({'sent_idx':i, 'fatcat_idx':sim_scores_max_sorted_idx[j], 'orig_sim_score':sim_scores_max[sim_scores_max_sorted_idx[j]],
     'orig_data':'nonsif', 'sim_score':0, 'sentence':df_nonsif1_input.iloc[i,0], 'fatcat-sentence':df_fatcat.iloc[sim_scores_max_sorted_idx[j],0]},ignore_index = True)
    fatcat_nonsif_count[sim_scores_max_sorted_idx[j]]+=1  

  
  print(df_psif_output.head)
  fatcat_psif_count_sorted = sorted(fatcat_psif_count.items(), key=lambda item: item[1])
  fatcat_nonsif_count_sorted = sorted(fatcat_nonsif_count.items(), key=lambda item: item[1])
  print(fatcat_psif_count_sorted)
  print(fatcat_nonsif_count_sorted)
  print("Total positive sentences :" + str(sum(fatcat_psif_count.values())))
  print("Total negative sentences :" + str(sum(fatcat_nonsif_count.values())))

  random_seed=2
  print("Random seed : " + str(random_seed))
  df_psif_test, df_psif_dev, df_psif_train=np.split(df_psif_output.sample(frac=1, random_state=random_seed), [1000, 2000])

  df_psif_test.to_csv('eval_file1_replaced_fatcats_v001.tsv', sep='\t', index=False, header=None) 
  df_psif_dev.to_csv('eval_file2_replaced_fatcats_v001.tsv', sep='\t', index=False, header=None) 

def compute_random_subset_accuracy():
  start_time = time.time()
  print("Start...")
  df_psif=pd.read_csv('model4_test_results_replaced_fatcats_psif.tsv',sep='\t',dtype=str)  
  df_nonsif=pd.read_csv('model4_test_results_replaced_fatcats_nonsif.tsv',sep='\t',dtype=str)

  # df_psif=pd.read_csv('model3_test_results_psif.tsv',sep='\t',dtype=str)  
  # df_nonsif=pd.read_csv('model3_test_results_nonsif.tsv',sep='\t',dtype=str)
  sentences=[]
  threshold=0.5
  psif_predictions=0
  nonsif_predictions=0
  # for i in range(df_input.shape[0]):#df.shape[0]):
  #   scores=df_input.iloc[i,2].strip('][').split(',')
  #   scores=[float(x) for x in scores]
  #   score_count=sum([x>=threshold for x in scores])
  #   if (score_count>0) and df_input.iloc[i,1]=='psif':
  #     psif_predictions=psif_predictions+1
  #   if (score_count==0) and df_input.iloc[i,1]=='nonsif':
  #     nonsif_predictions=nonsif_predictions+1
        
  # print(psif_predictions/4071)  
  # print(nonsif_predictions/2590)
    #print(scores)
  #print(sentences)
  #df_psif_train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
  r_state=102
  sub_set=500
  df_psif_test, df_psif_train=np.split(df_psif.sample(frac=1), [5000])
  df_nonsif_test, df_nonsif_train=np.split(df_nonsif.sample(frac=1), [5000])
  print(df_psif_test.shape)
  print(df_psif_train.shape)
  print(df_nonsif_test.shape)
  print(df_nonsif_train.shape)
  

  for i in range(df_psif_test.shape[0]):#df.shape[0]):
    score_max=float(df_psif_test.iloc[i,2])
    if (score_max>threshold) and df_psif_test.iloc[i,1]=='psif':
      psif_predictions=psif_predictions+1
  for i in range(df_nonsif_test.shape[0]):#df.shape[0]):
    score_max=float(df_nonsif_test.iloc[i,2])  
    if (score_max<=threshold) and df_nonsif_test.iloc[i,1]=='nonsif':
      nonsif_predictions=nonsif_predictions+1
        
  print(psif_predictions/df_psif_test.shape[0])  
  print(nonsif_predictions/df_nonsif_test.shape[0])
  print(((psif_predictions/df_psif_test.shape[0])+(nonsif_predictions/df_nonsif_test.shape[0]))/2) 
  
        
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))  

def compute_accuracy(output_file):
  start_time = time.time()
  print("Start...")
  df_input=pd.read_csv(output_file,sep='\t',dtype=str)  
  df_output=pd.DataFrame()
  sentences=[]
  threshold=0.5
  psif_predictions=0
  nonsif_predictions=0
  # for i in range(df_input.shape[0]):#df.shape[0]):
  #   scores=df_input.iloc[i,2].strip('][').split(',')
  #   scores=[float(x) for x in scores]
  #   score_count=sum([x>=threshold for x in scores])
  #   if (score_count>0) and df_input.iloc[i,1]=='psif':
  #     psif_predictions=psif_predictions+1
  #   if (score_count==0) and df_input.iloc[i,1]=='nonsif':
  #     nonsif_predictions=nonsif_predictions+1
        
  # print(psif_predictions/4071)  
  # print(nonsif_predictions/2590)
    #print(scores)
  #print(sentences)


  for i in range(df_input.shape[0]):#df.shape[0]):
    score_max=float(df_input.iloc[i,3])
    if (score_max>threshold) and df_input.iloc[i,1]=='psif':
      psif_predictions=psif_predictions+1
    if (score_max<=threshold) and df_input.iloc[i,1]=='nonsif':
      nonsif_predictions=nonsif_predictions+1
        
  print(psif_predictions/4071)  
  print(nonsif_predictions/2590)
  print(((psif_predictions/4071)+(nonsif_predictions/2590))/2) 
  
        
  print(psif_predictions/4071)  
  print(nonsif_predictions/2590)
  print(((psif_predictions/4071)+(nonsif_predictions/2590))/2)  
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))

def compute_ensemble_accuracy(input_file1,input_file2,input_file3):
  start_time = time.time()
  print("Start...")
  df_input1=pd.read_csv(input_file1,sep='\t',dtype=str)  
  df_input2=pd.read_csv(input_file2,sep='\t',dtype=str)  
  df_input3=pd.read_csv(input_file3,sep='\t',dtype=str)  
  print("Input loaded")

  sentences=[]
  threshold=0.5
  psif_predictions_mean=0
  nonsif_predictions_mean=0
  psif_predictions_max=0
  nonsif_predictions_max=0
  psif_predictions_median=0
  nonsif_predictions_median=0  
  psif_predictions_mean2=0
  nonsif_predictions_mean2=0
  psif_predictions_max2=0
  nonsif_predictions_max2=0  
  for i in range(df_input1.shape[0]):#df_input1.shape[0]
    print("Processing row:  "+str(i))
    scores1=df_input1.iloc[i,2].strip('][').split(',')
    scores1=[float(x) for x in scores1]
    scores2=df_input2.iloc[i,2].strip('][').split(',')
    scores2=[float(x) for x in scores2]
    scores3=df_input3.iloc[i,2].strip('][').split(',')
    scores3=[float(x) for x in scores3]
    
    scores_mean=[statistics.mean(x) for x in zip(scores1,scores2,scores3)]
    scores_max=[max(x) for x in zip(scores1,scores2,scores3)]
    scores_median=[statistics.median(x) for x in zip(scores1,scores2,scores3)]
    scores_mean2=[statistics.mean(x) for x in zip(scores1,scores3)]
    scores_max2=[max(x) for x in zip(scores1,scores3)]

    score_count=sum([x>=threshold for x in scores_mean])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_mean=psif_predictions_mean+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_mean=nonsif_predictions_mean+1
        
    score_count=sum([x>=threshold for x in scores_max])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_max=psif_predictions_max+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_max=nonsif_predictions_max+1

    score_count=sum([x>=threshold for x in scores_median])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_median=psif_predictions_median+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_median=nonsif_predictions_median+1      

    score_count=sum([x>=threshold for x in scores_mean2])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_mean2=psif_predictions_mean2+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_mean2=nonsif_predictions_mean2+1
        
    score_count=sum([x>=threshold for x in scores_max2])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_max2=psif_predictions_max2+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_max2=nonsif_predictions_max2+1  


  # for i in range(df_input.shape[0]):#df.shape[0]):
  #   score_max=float(df_input.iloc[i,3])
  #   if (score_max>threshold) and df_input.iloc[i,1]=='psif':
  #     psif_predictions=psif_predictions+1
  #   if (score_max<=threshold) and df_input.iloc[i,1]=='nonsif':
  #     nonsif_predictions=nonsif_predictions+1
        
  print(psif_predictions_mean/4071)  
  print(nonsif_predictions_mean/2590)
  print(((psif_predictions_mean/4071)+(nonsif_predictions_mean/2590))/2) 

  print(psif_predictions_max/4071)  
  print(nonsif_predictions_max/2590)
  print(((psif_predictions_max/4071)+(nonsif_predictions_max/2590))/2) 

  print(psif_predictions_median/4071)  
  print(nonsif_predictions_median/2590)
  print(((psif_predictions_median/4071)+(nonsif_predictions_median/2590))/2)   

  print(psif_predictions_mean2/4071)  
  print(nonsif_predictions_mean2/2590)
  print(((psif_predictions_mean2/4071)+(nonsif_predictions_mean2/2590))/2) 

  print(psif_predictions_max2/4071)  
  print(nonsif_predictions_max2/2590)
  print(((psif_predictions_max2/4071)+(nonsif_predictions_max2/2590))/2)  
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))  


def find_error_sentences(test_file, input_file1,input_file2,input_file3):
  start_time = time.time()
  print("Start...")
  df_input1=pd.read_csv(input_file1,sep='\t',dtype=str)  
  df_input2=pd.read_csv(input_file2,sep='\t',dtype=str)  
  df_input3=pd.read_csv(input_file3,sep='\t',dtype=str)  
  print("Input loaded")

  seperator='\t' if test_file.split('.')[-1]=='tsv' else ','  
  df_output=pd.read_csv(test_file,sep=seperator,dtype=str)
  sentences=[]
  threshold=0.5
  psif_predictions_mean=0
  nonsif_predictions_mean=0
  psif_predictions_max=0
  nonsif_predictions_max=0
  psif_predictions_median=0
  nonsif_predictions_median=0  
  psif_predictions_mean2=0
  nonsif_predictions_mean2=0
  psif_predictions_max2=0
  nonsif_predictions_max2=0  
  for i in range(100):#df_input1.shape[0]
    print("Processing row:  "+str(i))
    scores1=df_input1.iloc[i,2].strip('][').split(',')
    scores1=[float(x) for x in scores1]
    scores2=df_input2.iloc[i,2].strip('][').split(',')
    scores2=[float(x) for x in scores2]
    scores3=df_input3.iloc[i,2].strip('][').split(',')
    scores3=[float(x) for x in scores3]
    
    scores_mean=[statistics.mean(x) for x in zip(scores1,scores2,scores3)]
    scores_max=[max(x) for x in zip(scores1,scores2,scores3)]
    scores_median=[statistics.median(x) for x in zip(scores1,scores2,scores3)]
    scores_mean2=[statistics.mean(x) for x in zip(scores1,scores3)]
    scores_max2=[max(x) for x in zip(scores1,scores3)]

    score_count=sum([x>=threshold for x in scores_mean])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_mean=psif_predictions_mean+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_mean=nonsif_predictions_mean+1
        
    score_count=sum([x>=threshold for x in scores_max])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_max=psif_predictions_max+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_max=nonsif_predictions_max+1

    score_count=sum([x>=threshold for x in scores_median])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_median=psif_predictions_median+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_median=nonsif_predictions_median+1      

    score_count=sum([x>=threshold for x in scores_mean2])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_mean2=psif_predictions_mean2+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_mea2n=nonsif_predictions_mean2+1
        
    score_count=sum([x>=threshold for x in scores_max2])
    if (score_count>0) and df_input1.iloc[i,1]=='psif':
      psif_predictions_max2=psif_predictions_max2+1
    if (score_count==0) and df_input1.iloc[i,1]=='nonsif':
      nonsif_predictions_max2=nonsif_predictions_max2+1  


  # for i in range(df_input.shape[0]):#df.shape[0]):
  #   score_max=float(df_input.iloc[i,3])
  #   if (score_max>threshold) and df_input.iloc[i,1]=='psif':
  #     psif_predictions=psif_predictions+1
  #   if (score_max<=threshold) and df_input.iloc[i,1]=='nonsif':
  #     nonsif_predictions=nonsif_predictions+1
        
  print(psif_predictions_mean/4071)  
  print(nonsif_predictions_mean/2590)
  print(((psif_predictions_mean/4071)+(nonsif_predictions_mean/2590))/2) 

  print(psif_predictions_max/4071)  
  print(nonsif_predictions_max/2590)
  print(((psif_predictions_max/4071)+(nonsif_predictions_max/2590))/2) 

  print(psif_predictions_median/4071)  
  print(nonsif_predictions_median/2590)
  print(((psif_predictions_median/4071)+(nonsif_predictions_median/2590))/2)   

  print(psif_predictions_mean2/4071)  
  print(nonsif_predictions_mean2/2590)
  print(((psif_predictions_mean2/4071)+(nonsif_predictions_mean2/2590))/2) 

  print(psif_predictions_max2/4071)  
  print(nonsif_predictions_max2/2590)
  print(((psif_predictions_max2/4071)+(nonsif_predictions_max2/2590))/2) 

  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))  


def train_model():
  start_time = time.time()
  print("Start...")
  model = SentenceTransformer('stsb-roberta-large')
  sts_reader = readers.STSBenchmarkDataReader('./', normalize_scores=False)
  train_data = SentencesDataset(sts_reader.get_examples('train_file_replaced_fatcats_v017.tsv'), model)#train_file_replaced_fatcats_v007  msr_paraphrase_sts_format_train
  train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
  train_loss = losses.CosineSimilarityLoss(model=model)
  evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('eval_file1_replaced_fatcats_v001.tsv'))
  model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=8,
          evaluation_steps=2000,
          warmup_steps=10000,
          output_path="./models/model_23")

  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds or {:.01f} minutes or or {:.01f} hours  '.format(elapsed_time,elapsed_time/60,elapsed_time/3600  ))  



if __name__ == '__main__':  

  #get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'psif_test_sentences_6661_rows_20DEC20.csv', 'replaced_fatcats_psif_test_output_01.tsv', 'paraphrase-distilroberta-base-v1')
  #get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'psif_test_sentences_6661_rows_20DEC20.csv', 'replaced_fatcats_psif_test_output_02.tsv', 'stsb-roberta-large')
  #get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'psif_test_sentences_6661_rows_20DEC20.csv', 'replaced_fatcats_psif_test_output_04.tsv', 'paraphrase-xlm-r-multilingual-v1')
  
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_psif_with_sim_max.tsv', 'trainset_filtered_psif_with_simscores_repfatcats_model1.tsv', 'paraphrase-distilroberta-base-v1')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_psif_with_sim_max.tsv', 'trainset_filtered_psif_with_simscores_repfatcats_model2.tsv', 'stsb-roberta-large')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_psif_with_sim_max.tsv', 'trainset_filtered_psif_with_simscores_repfatcats_model4.tsv', 'paraphrase-xlm-r-multilingual-v1')

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_nonsif_with_sim_max.tsv', 'trainset_filtered_nonsif_with_simscores_repfatcats_model1.tsv', 'paraphrase-distilroberta-base-v1')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_nonsif_with_sim_max.tsv', 'trainset_filtered_nonsif_with_simscores_repfatcats_model2.tsv', 'stsb-roberta-large')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'trainset_filtered_nonsif_with_sim_max.tsv', 'trainset_filtered_nonsif_with_simscores_repfatcats_model4.tsv', 'paraphrase-xlm-r-multilingual-v1')
  
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_with_simscores_repfatcats_model1.tsv', 'paraphrase-distilroberta-base-v1')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_with_simscores_repfatcats_model2.tsv', 'stsb-roberta-large')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_with_simscores_repfatcats_model4.tsv', 'paraphrase-xlm-r-multilingual-v1')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_with_simscores_repfatcats_model1.tsv', 'paraphrase-distilroberta-base-v1')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_with_simscores_repfatcats_model2.tsv', 'stsb-roberta-large')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_with_simscores_repfatcats_model4.tsv', 'paraphrase-xlm-r-multilingual-v1')
  # generate_dev_data()

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  #compute_test_set_accuracy('devset_filtered_psif_with_simscores_repfatcats_model16.tsv', 0.9) 

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_psif_with_sim_max.tsv', 'testset_filtered_psif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_nonsif_with_sim_max.tsv', 'testset_filtered_nonsif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  # compute_test_set_accuracy('testset_filtered_psif_with_simscores_repfatcats_model16.tsv', 0.8) 

  
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'filtered_psif.tsv', 'filtered_psif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'filtered_nonsif.tsv', 'filtered_nonsif_with_simscores_repfatcats_model16.tsv', './models/model_16')
  
  # compute_test_set_accuracy('filtered_psif_with_simscores_repfatcats_model16.tsv',0.6) 

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_psif_with_sim_max.tsv', 'testset_filtered_psif_with_simscores_repfatcats_model22.tsv', './models/model_22')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_nonsif_with_sim_max.tsv', 'testset_filtered_nonsif_with_simscores_repfatcats_model22.tsv', './models/model_22')
  # compute_test_set_accuracy('testset_filtered_psif_with_simscores_repfatcats_model22.tsv', 0.46) 

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_with_simscores_repfatcats_model22.tsv', './models/model_22')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_with_simscores_repfatcats_model22.tsv', './models/model_22')
  # compute_test_set_accuracy('devset_filtered_psif_with_simscores_repfatcats_model22.tsv', 0.48) 


  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_psif_with_sim_max.tsv', 'testset_filtered_psif_with_simscores_repfatcats_model23.tsv', './models/model_23')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_nonsif_with_sim_max.tsv', 'testset_filtered_nonsif_with_simscores_repfatcats_model23.tsv', './models/model_23')
  

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_repfatcats_model22.tsv', './models/model_22')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_repfatcats_model22.tsv', './models/model_22')

  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_psif_with_sim_max.tsv', 'testset_filtered_psif_repfatcats_model22.tsv', './models/model_22')
  # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_nonsif_with_sim_max.tsv', 'testset_filtered_nonsif_repfatcats_model22.tsv', './models/model_22')
  #compute_test_set_accuracy('testset_filtered_psif_with_simscores_repfatcats_model16.tsv', 0.9) 

 # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_psif_with_sim_max.tsv', 'testset_filtered_psif_with_simscores_repfatcats_model23.tsv', './models/model_23')
 # get_similarity_scores('replaced_fatcats_with_index_20DEC20.tsv', 'testset_filtered_nonsif_with_sim_max.tsv', 'testset_filtered_nonsif_with_simscores_repfatcats_model23.tsv', './models/model_23')
 
  #generate_predictions_file('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_psif_with_sim_max.tsv', 'devset_filtered_psif_repfatcats_with_fatcat_sentence_model23.tsv', './models/model_23')
  generate_predictions_file('replaced_fatcats_with_index_20DEC20.tsv', 'devset_filtered_nonsif_with_sim_max.tsv', 'devset_filtered_nonsif_repfatcats_with_fatcat_sentence_model23.tsv', './models/model_23')


  # psif_acc_array=[]
  # non_sif_acc_array=[]
  # avg_acc_array=[]
  # acc_range=np.arange(0.3,0.9,0.05)
  # for th in acc_range:
  #   psif_acc,non_sif_acc, avg_acc= compute_test_set_accuracy('devset_filtered_psif_with_simscores_repfatcats_model22.tsv', th) 
  #   psif_acc_array.append(psif_acc)
  #   non_sif_acc_array.append(non_sif_acc)
  #   avg_acc_array.append(avg_acc)

  # fig, axs = plt.subplots(1, 3)
  # axs[0].plot(acc_range, psif_acc_array,'-*')
  # axs[0].grid(True)
  # axs[1].plot(acc_range, non_sif_acc_array,'-*')
  # axs[1].grid(True)
  # axs[2].plot(acc_range, avg_acc_array,'-*')
  # axs[2].grid(True)
  # plt.show()
  # plt.savefig('1.jpg')  

  #compute_filtered_data_accuracy()


  # #compute_accuracy('replaced_fatcats_psif_test_output_01.tsv')
  #compute_accuracy('replaced_fatcats_psif_test_output_02.tsv')
  #compute_accuracy('replaced_fatcats_psif_test_output_04.tsv')
  #compute_accuracy('psif_test_output_01.tsv')
  #compute_ensemble_accuracy('psif_test_output_01.tsv','psif_test_output_02.tsv','psif_test_output_04.tsv')
  #compute_ensemble_accuracy('replaced_fatcats_psif_test_output_01.tsv','replaced_fatcats_psif_test_output_02.tsv','replaced_fatcats_psif_test_output_04.tsv')
  #'psif_test_sentences_6661_rows_20DEC20.csv'
  #generate_test_set('replaced_fatcats_psif_test_output_04.tsv')
  #compute_random_subset_accuracy()
  #generate_filtered_test_set()
  #compute_filtered_data_accuracy()
  #generate_train_dev_test_set()
  #generate_training_data()
  #train_model()
  
     


  