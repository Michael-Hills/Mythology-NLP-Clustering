import numpy as np
import tqdm
import random
import pandas as pd
from hyperopt import fmin,space_eval,partial,Trials,tpe,STATUS_OK,hp
import seaborn as sns
import matplotlib.pyplot as plt
from clustering import *

def plotResults():
    """
    Function to plot the pairwise relaationships between parameters
    """

    df = pd.read_csv('results2.csv')
    df = df.drop(columns=['run_id'])
    sns.set(rc={'figure.figsize':(5,5)})
    sns.pairplot(df,hue='label_count')
    plt.show()
    return


def scoreClusters(clusters, threshold):
    """
    Function to calculate the cluster confidence cost
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    cost = (np.count_nonzero(clusters.probabilities_ < threshold)/len(cluster_labels))
    
    return label_count, cost

def randomSearch(data, space, evals):
    """
    Function to perform a random seach within a specified search space
    """
    
    results = []
    
    for i in tqdm.trange(evals):
        n_neighbors = random.choice(space['n_neighbors'])
        min_dist = random.choice(space['min_dist'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        min_samples = random.choice(space['min_samples'])
        
        clusters = generateClusters( data,
                                     n_neighbors = n_neighbors, 
                                     min_dist=min_dist,
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size,
                                     min_samples=min_samples,
                                     )
    
        label_count, cost = scoreClusters(clusters, threshold = 0.05)
                
        results.append([i, n_neighbors, min_dist,n_components, min_cluster_size, min_samples,
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'min_dist','n_components', 
                                               'min_cluster_size','min_samples', 'label_count', 'cost'])
    
    
    
    return result_df.sort_values(by='cost')

def getRandomScores(data):
    """
    Function that runs the random search
    """  

    pd.set_option('display.max_rows', None)
    space = {
        "n_neighbors": range(5,100,5),
        "min_dist": np.arange(0,0.14,0.02),
        "n_components": range(3,10),
        "min_cluster_size": range(5,35,5),
        "min_samples": range(1,8)}

    randomScores = randomSearch(data,space,300)

    print(randomScores.head(200))


def objective(params, embeddings, lower, upper):
    """
    Objective function to minimise
    """
    
    clusters = generateClusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 min_dist = params['min_dist'],
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 min_samples = params['min_samples'])
    
    label_count, cost = scoreClusters(clusters, threshold = 0.05)
    
    #10% penalty if too many or too little clusters
    if (label_count < lower) | (label_count > upper):
        penalty = 0.10

    else:
        penalty = 0

    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesianSearch(embeddings, space, lower, upper, maxEvals):
    """
    Function to perform Bayesian search by minimising thee objective function
    """
    
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, lower=lower, upper=upper)
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=maxEvals, 
                trials=trials)

    best = space_eval(space, best)
    print ('best:')
    print (best)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    bestClusters = generateClusters(embeddings, 
                                      n_neighbors = best['n_neighbors'], 
                                      min_dist = best['min_dist'],
                                      n_components = best['n_components'], 
                                      min_cluster_size = best['min_cluster_size'],
                                      min_samples = best['min_samples'])
    
    return best, bestClusters, trials

def getBayesianScores(data):
    """
    Function that runs the bayesian search
    """  

    pd.set_option('display.max_rows', None)
    hspace = {
        "n_neighbors": hp.choice('n_neighbors',range(2,32,2)),
        "min_dist":hp.choice('min_dist',np.arange(0,0.14,0.02)),
        "n_components": hp.choice('n_components',range(5,11)),
        "min_cluster_size": hp.choice('min_cluster_size',range(10,40,5)),
        "min_samples": hp.choice('min_samples',range(1,6))}

    best_params_use, best_clusters_use, trials_use = bayesianSearch(data,space=hspace,lower=35,upper=150,maxEvals=150)

    