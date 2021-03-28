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
import torch

print("Is CUDA available : " + str(torch.cuda.is_available()))
print("GPU devices : " + str(torch.cuda.device_count()))
if torch.cuda.device_count()>1:
  print("GPU current device : " + torch.cuda.current_device())
  print("GPU device name : " + torch.cuda.get_device_name(torch.cuda.current_device()))


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def generate_predictions_file(fatcats_input_file, test_input_file, output_file, model_name):
  """Generates predictions file
  Args:
    fatcats_input_file: Input file in the same format as fatcats_with_index_20DEC20.csv
    test_input_file: Input file with test sentences that needs to be predicted (It should have the same format as psif_test_sentences_6661_rows_20DEC20.csv)
    output_file: name of output file 
    model_name: path to unzipped model folder

  Returns:
    output_file: file containing predictions
  """
  start_time = time.time()
  print("Start...") 
  seperator='\t' if fatcats_input_file.split('.')[-1]=='tsv' else ','
    
  df=pd.read_csv(fatcats_input_file,sep=seperator,dtype=str)  
  sentences=[]
  for i in range(df.shape[0]):#df.shape[0]):
    sentences.append(df.iloc[i,0])
  print("Starting embedding generation... This could take a while depending on CPU or GPU...") 
  model = SentenceTransformer(model_name)  
  sentence_embeddings = model.encode(sentences)
  print(sentence_embeddings[1].shape)
  elapsed_time = time.time() - start_time
  print('Took {:.03f} seconds for ref sentence embeddings'.format(elapsed_time)) 

  seperator='\t' if test_input_file.split('.')[-1]=='tsv' else ','  
  df_input=pd.read_csv(test_input_file,sep=seperator,dtype=str)  
  sim_scores, fatcat_sentence, sim_max, sim_min, sim_max_index, sim_min_index, ground_truth = [], [], [], [], [], [], []

  for i in range(df_input.shape[0]):#df_input.shape[0]):
    if (i%100==0):
      print("Example : " + str(i))   
      elapsed_time = time.time() - start_time
      print('{:.03f} seconds since start'.format(elapsed_time)) 
    query = df_input.iloc[i,0]
    query_vec = model.encode([query])[0]
    sim=[]
    for ref_idx in range(len(sentence_embeddings)):
      sim.append(cosine(query_vec,sentence_embeddings[ref_idx]))
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

  offset= 1 if(df_input.shape[1]<1) else 0
  df_input.insert(2-offset, "fatcat_sentence", fatcat_sentence)
  df_input.insert(3-offset, "prediction", sim_max)
  df_input.insert(4-offset, "ground_truth", ground_truth)
  df_input.insert(5-offset, "fatcat_sentence_index", sim_max_index)
  
  df_input.to_csv(output_file, sep='\t', index=False)   
  elapsed_time = time.time() - start_time 
  print('Took {:.03f} seconds'.format(elapsed_time))   


if __name__ == '__main__':  
  generate_predictions_file('replaced_fatcats_with_index_20DEC20.tsv', 'input_psif.tsv', 'output_psif_model23.tsv', './models/model_23')
  generate_predictions_file('replaced_fatcats_with_index_20DEC20.tsv', 'input_nonsif.tsv', 'output_nonsif_model23.tsv', './models/model_23')

