import umap.umap_ as umap
import hdbscan
import networkx as nx
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#from pyclustertend import ivat,hopkins
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import spacy
import numpy as np
import json

"""
def clusterTendency(data):

    print("Hopkins Value: ", hopkins(data,len(data)))

    ivat(data)
    plt.show()"""

def generateClusters(data,n_neighbors,min_dist,n_components,min_cluster_size,min_samples):
    """
    Function to create clusters
    """

    #reduce dimensions with UMAP
    clusterFit = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=83,
                metric='cosine'
        )
    
    reduced = clusterFit.fit_transform(data)
    
    #cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,gen_min_span_tree=True).fit(reduced)

    return clusterer
            
    
def visualiseClusters(data,clusterer,n_neighbors,min_dist):
    """
    Function to show the clusters in 2D space and the tree heirarchy
    """

    
    visualisationFit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=83,
            metric='cosine'
        )

    visualise = visualisationFit.fit_transform(data)
                    
    fig2 = px.scatter(visualise, x=0, y=1, color = clusterer.labels_)
    fig2.update_traces(mode="markers")
    fig2.show()
                
    clusterer.condensed_tree_.plot(select_clusters=True,selection_palette=sns.color_palette('deep', 20))
    plt.show()


def TFIDF(fileName,clusterer):
    """
    Function to perform TF-IDF or count vectorisation
    """

    #get the vectors and text from file
    df = pd.read_json(fileName, lines=True)
    df['clusterLabel'] = clusterer.labels_
    df['shortText'] = df['text'].str[0:1800]
    noClusters = df['clusterLabel'].max() + 1

    tfidf_vectorizer = CountVectorizer(input='content', stop_words='english')
    nlp = spacy.load('en_core_web_sm')
    included_tags = {"NOUN", "ADJ","VERB"}
    clusterTexts = []
    for i in range(noClusters):
        newText = []
        pages = df.loc[df['clusterLabel'] == i]
        pages = pages['shortText'].values.tolist()
        pages = ' '.join(pages)
        #remove all words that arent nouns, adjectives or verbs
        for token in nlp(pages):
            if token.pos_ in included_tags:
                newText.append(token.text)

        text = " ".join(newText)
        clusterTexts.append(text)
   


    labels = []
    titles = []

    #get the top 7 words for all clusters
    tfidf_vector = tfidf_vectorizer.fit_transform(clusterTexts)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    transposed = tfidf_df.T

    for i in range(noClusters):
        labels.append(transposed.sort_values(by=[i],ascending=False).head(7).index.tolist())
        titles.append(df.loc[df['clusterLabel'] == i]['title'].tolist())


    return titles,labels

def getCategories():    
    """
    Function to find the cultural region from the wiki categories
    """  
    
    mythologyByArea = {  "African mythology": ["Berber mythology","Yoruba mythology","Ashanti mythology"],
                            "Australian mythology":["Australian Aboriginal mythology","Māori mythology",'Aboriginal gods'],
                            "Asian mythology":["Buddhist mythology","Elamite mythology","Indonesian mythology","Korean mythology",
                                         "Philippine mythology","Tibetan mythology","Philippine creatures",'Buddhist demons','Buddhist folklore',
                                         'Philippine demons'],
                            "European mythology":["Baltic mythology","Basque mythology",'German mythology',
                                            "Christian mythology","Ars Goetia Demons","Dacian mythology","Etruscan mythology","Finnish mythology",
                                            "French mythology","Galician mythology","Germanic mythology","Portuguese mythology",'Etruscan goddesses',
                                            "Romanian mythology","Slavic mythology","Spanish mythology","Christian folklore",
                                            "Danish folklore","French folklore","German folklore","Icelandic folklore","Norwegian folklore",
                                            "Scandinavian folklore","Swedish folklore","Nymphs",'European dragons','Medieval European creatures',
                                            'French creatures','Germanic weapons','Armenian mythology','Christian demons','Germanic heroic legends',
                                            'Dutch mythology'],
                            "Middle Eastern mythology":["Islamic mythology","Jewish mythology","Arabian mythology","Levantine mythology",
                                                  "Magian mythology","Persian mythology","Semitic mythology", "Maltese folklore","Islamic demons"],
                            "Chinese mythology":['Chinese creatures'],
                            "Japanese mythology":["Japanese folklore","Japanese weapons","Japanese items",'Kitsune'],
                            "Indian mythology":["Hindu mythology","Women in Hindu mythology",'Hindu goddesses'],
                            "Celtic mythology":["Galician mythology","Manx folklore","Irish mythology","Irish gods","Irish kings","Irish folklore",
                                                'Celtic goddesses','Celtic gods'],
                            "Egyptian mythology":['Egyptian creatures'],
                            "British mythology":["English mythology","Scottish mythology","Welsh mythology","British folklore","English folklore",
                                                "Manx folklore","Scottish folklore",'English creatures','Arthurian legends'],
                            "Mesopotamian mythology":["Mesopotamian Mythology",'Ishtar'],
                            "Indigenous American mythology":["Aztec mythology","Native American mythology","Native American folklore"],
                            "North American mythology":["Cajun mythology","Caribbean mythology","Hawaiian mythology","Hopi mythology",
                                                        "Inuit mythology","Maya mythology","Mexican mythology","Māori mythology",
                                                        "Polynesian mythology","Oceanian mythology","Zuni mythology","American folklore"],
                            "South American mythology":["Aztec mythology","Brazilian mythology","American folklore","Brazilian folklore",
                                                        "Latin American folklore","South American folklore"],
                            "Classical mythology":[],
                            "Greek mythology":['Greek creatures',"Greek people", "Greek figures","Greek heroes","Greek goddesses","Greek gods",
                                               "Greek items",'Greek deities'
                                               "Greek monarchs","Demigods from Greek myths","Peoples in Greek mythology","Achaeans","Daemons",
                                               "Heracles",'Iliad characters', 'Trojans','Greek locations','Greek events','Twelve Olympians',
                                               'Mortal demigods from Greek myths','Mortals from Greek myths','Greek poems','Greek giants',
                                               'Greek monarchs'],
                            "Roman mythology":['Roman creatures'],
                            "Norse mythology":['Æsir','Norse figures','Norse weapons','Norse locations', 'Norse realms','Norse items',
                                               'Old Norse studies scholars','Runes']
    }

    df = pd.DataFrame.from_dict(mythologyByArea,orient='index')
    df = df.transpose()
    
    #get categories from file
    categoriesDf = pd.read_json('textVectors.jsonl', lines=True)
    categories = categoriesDf['categories'].values.tolist()
    areas = []
    for category in categories:
        mythArea = set()
        for i in category:
            if i in df.columns:
                mythArea.add(i)
            else:
                for col in df.columns:
                    if i in df[col].values:
                        if col not in mythArea:
                            mythArea.add(col)

        if ("Classical mythology" in mythArea and "Roman mythology" in mythArea) or ("Classical mythology" in mythArea and 
                                                                                    "Greek mythology" in mythArea):
            mythArea.remove("Classical mythology")

        areas.append(list(mythArea))

    #add regions to json
    copy = categoriesDf.copy()
    copy['area'] = areas
    copy.dropna(subset=['textVector'],inplace=True)
    copy.to_json('textVectors.jsonl',orient='records', lines = True)

    

def calculateProportions(clusterer):

    """
    Function to calculate the cultural proportions
    """  
    df = pd.read_json('textVectors.jsonl', lines=True)
    df['clusterLabel'] = clusterer.labels_
    areas = {}
    for i in range(max(clusterer.labels_)+1):
        clusterAreas = []
        pages = df.loc[df['clusterLabel'] == i]
        pages = pages['area'].values.tolist()
        for page in pages:
            clusterAreas.extend(page)

        #count each occurance
        areas[i] = dict(Counter(clusterAreas))

    return areas



        
