from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from clustering import *
import numpy as np
from score import *
from clustering import *
import networkx as nx
import jsonlines


def getVectors(model):
    """
    Function to get the sentence embedding vectors from each page
    """

    #return dataframe with the vectors and text if already calculated
    if(os.path.isfile(os.getcwd() + "/textVectors.jsonl")):
        df = pd.read_json('textVectors.jsonl', lines=True)
        print("Vectors loaded from JSON Lines File")
        return df

    #otherwise calculate the vectors
    else:

        vectors = []
        df = pd.read_json('data.jsonl', lines=True)
        textdf = df[['title','text','categories']]
        texts = textdf['text'].values.tolist()

        copy = textdf.copy()

        for text in texts:
            vectors.append(model.encode(text))
        
        copy['textVector'] = vectors
        copy.dropna(subset=['textVector'],inplace=True)
        copy.to_json('textVectors.jsonl',orient='records', lines = True)

        print("Vectors created and written to JSON Lines file")
        return copy


def cluster(vectors,parameterSearch=False,visualise=False):
    """
    Function to cluster the vectors
    """
    

    data = vectors['textVector'].values.tolist()
    if (parameterSearch == True):
        results = getRandomScores(data)

    neighbors = 5
    dist = 0
    
    clusterer = generateClusters(data,neighbors,dist,5,10,1)
    print("Cluster Sizes: ", Counter(clusterer.labels_))
    print("No. of Clusters: ", max(clusterer.labels_)+1)
    print("Cluster Score: " ,scoreClusters(clusterer,threshold = 0.05))

    
    titles, labels = TFIDF('textVectors.jsonl',clusterer)
    areas = calculateProportions(clusterer)

    print(titles[0])
    print(labels[0])
    print(areas[0])

    """with jsonlines.open('clusterResults.jsonl', mode='a') as writer:
        for i in range(max(clusterer.labels_)+1):
            dict = {"titles": titles[i],"labels": labels[i],"areas":areas[i]}
            writer.write(dict)"""
               
    if (visualise == True):
        visualiseClusters(data,clusterer,neighbors,dist)




    
  
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = getVectors(model)
cluster(vectors)






