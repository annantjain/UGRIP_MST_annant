from datasets import load_dataset, Dataset,get_dataset_config_names,load_from_disk, Dataset, DatasetDict, concatenate_datasets
import random
import numpy as np
import random
import torch
from sklearn.utils import resample
import pandas as pd
import re

def datamaker(setlist, sz, lang,langlong,langfull):
  train_lang = []
  ddlang = []
  listlen = 0
  for s in setlist:
    print("Processing: " + s)
    if (lang in get_dataset_config_names('mbzuai-ugrip-statement-tuning/'+s)) or (langlong in get_dataset_config_names('mbzuai-ugrip-statement-tuning/'+s)) or (langfull in get_dataset_config_names('mbzuai-ugrip-statement-tuning/'+s)):
      if s == "Topic-Statements" or s == "belebele":
        dataset = load_dataset('mbzuai-ugrip-statement-tuning/'+s, langfull, split='train', streaming=True)
      elif s == "sentiments":
        dataset = load_dataset('mbzuai-ugrip-statement-tuning/'+s, langlong, split='train', streaming=True)
      else:
        dataset = load_dataset('mbzuai-ugrip-statement-tuning/'+s, lang, split='train', streaming=True)
      train_lang.append(dataset.take(sz))
      #listlen += 1
      print("Finished: "+s)
    else:
      print(lang+" not found in " + s)
      pass
      
  print("Starting Merging")

  for i in train_lang:
    ds = Dataset.from_generator(lambda: (yield from i), features=i.features)
    dd = DatasetDict({"train": ds})
    ddlang.append(dd)
  trainset = ddlang[0]["train"]
  for i in range(1,len(ddlang)-1):
    trainset = concatenate_datasets([trainset,ddlang[i]["train"]])
  return DatasetDict({"train": trainset})

#Wrapper Function for datamaker
def fullmaker(length, langlist):
  langdata = []
  for lang in langlist:
    langshort,langfull,langlong = lang
    print("-----------------Gathering: " + langlong)
    data = datamaker(sets,length,langshort,langfull,langlong)
    langdata.append(data)

  print("--------------------------------Making Final Dataset----------------------------------------")
  finalset = langdata[0]["train"]
  for i in range(1,len(langdata)-1):
    finalset = concatenate_datasets([finalset,langdata[i]["train"]])
  
  finaldataset = []
  for i in range(len(finalset)):
    thing = {}
    if finalset[i]["statement"] != None:
      thing["Text"] = finalset[i]["statement"]
    elif finalset[i]["text"] != None:
      thing["Text"] = finalset[i]["text"]
    else:
      print("missing text?",i)

    if finalset[i]["is_true"] != None:
      thing["label"] = finalset[i]["is_true"]
    elif finalset[i]["label"] != None:
      thing["label"] = finalset[i]["label"]
    else:
      print("missing label?",i)
    
    finaldataset.append(thing)
  return finaldataset



sets = ["xquad", "xlwic", "massive", "wikilingual_dataset","paws-x","Topic-Statements","belebele","sentiments","exams"]
langlist = [("zh","chinese","zho_Hans"), ("en","english","eng_Latn"),("fr","french","fra_Latn"),("vi","vietnamese","vie_Latn")]

finalset = fullmaker(850,langlist)
