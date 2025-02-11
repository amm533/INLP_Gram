import directions

import matplotlib.pyplot as plt
import numpy as np 
from numpy import dot
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
import math
import random
from itertools import combinations, filterfalse
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def Grammatical_Gender_Classification_Test(gender_pairs, model, k):
    """
    Test for checking that gendered words can be classified by grammatical gender.
    :param gender_pairs: the pairs of words (feminine word, masculine word) 
    :param model: 
    :param k: the number of executions with a different random state of the classifier
    :return: 
    """
    
    f_vecs = []
    m_vecs = []

    n_f = 0
    n_m = 0

    for pair in gender_pairs:
        if pair[0] in model:
            f_vecs.append(model[pair[0]])
            n_f = n_f + 1
        
        if pair[1] in model:    
            m_vecs.append(model[pair[1]])
            n_m = n_m + 1
    
    X = f_vecs + m_vecs

    y = [0] * n_f + [1] * n_m
       


    print()
    print("Logistic Regression")
    print("_________")   

    acc_k = []
    
    for random_i in range(1,k+1):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_i)
        
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        acc = logreg.score(X_test, y_test)
        acc_k.append(acc)

    logreg_mean_acc = np.mean(acc_k)
    logreg_std_dev_acc = np.std(acc_k) # standard deviation, a measure of the spread of a distribution

    print(f"{k} random seeds")
    print(f"Mean Accuracy: {100*logreg_mean_acc}%")
    print(f"All accuracies: {acc_k}")
    print(f"Mean Standard Deviation: {100*logreg_std_dev_acc}%")

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, k+1), acc_k, marker='o', linestyle='-', color='b')
    plt.axhline(y=logreg_mean_acc, color='r', linestyle='--', label=f'Mean Accuracy: {100*logreg_mean_acc:.2f}%')
    plt.title(f"Accuracy Evolution Across {k} Random Seeds")
    plt.xlabel('Iteration (Random Seed)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.5)  # Accuracy is between 0 and 1
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_on_gender_direction(professions, model, gender_direction):
    """
    Given a list of professions of the form [[singular F, singular M, plural M, plural F],[],...],ç
    generate a plot of projections onto the gender direction
    :param professions: 
    :param model: 
    :param gender_direction: 
    """
    x_coords = []
    y_coords = []
    i = 0
        
    word_labels = []
    
    avg_sg_m = 0 # Media de singulares m 
    avg_sg_f = 0 # Media de singulares f
    avg_pl_m = 0 # Media de plurales m
    avg_pl_f = 0 # Media de plurales m
    
    for profession in professions:

        if profession[0] not in model or profession[1] not in model or profession[2] not in model or profession[3] not in model:
            continue # ignore the pair

        y_coords.append(i)
        y_coords.append(i)
        y_coords.append(i)
        y_coords.append(i)
        i = i+1

        # Projections
        proy_sg_f = model[profession[0]].dot(gender_direction)
        proy_sg_m = model[profession[1]].dot(gender_direction)
        proy_pl_m = model[profession[2]].dot(gender_direction)
        proy_pl_f = model[profession[3]].dot(gender_direction)
        
        x_coords.append(proy_sg_f)
        word_labels.append(profession[0])      
        
        x_coords.append(proy_sg_m)
        word_labels.append(profession[1])

        x_coords.append(proy_pl_m)
        word_labels.append(profession[2])

        x_coords.append(proy_pl_f)
        word_labels.append(profession[3])

        avg_sg_m = avg_sg_m + proy_sg_m
        avg_sg_f = avg_sg_f + proy_sg_f
        avg_pl_m = avg_pl_m + proy_pl_m
        avg_pl_f = avg_pl_f + proy_pl_f

    avg_sg_m = avg_sg_m/i
    avg_sg_f = avg_sg_f/i
    avg_pl_m = avg_pl_m/i
    avg_pl_f = avg_pl_f/i
    
    fig=plt.figure(figsize=(10, 7), dpi=80)
    plt.scatter(x_coords, y_coords, marker='x')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):

        if (k+1)%4==0: 
            color = 'purple' # plural femenine
            
        elif (k+2)%4==0:
            color = 'green' # plural masculine
            
        elif (k+3)%4==0:
            color = 'blue' # singular masculine
            
        else: 
            color ='red' # singular femenine

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=8,
                     color=color)

    plt.xlim(-2, 2)
    plt.ylim(0, 40)

    plt.axvline(x=0, color='k', linestyle='--', label='Neutralidad (proy=0) de género semántico')    
    plt.axvline(x=avg_sg_m, color='blue', linestyle='--', label=f'Average singular masculine: {avg_sg_m}')
    plt.axvline(x=avg_sg_f, color='red', linestyle='--', label=f'Average singular feminine: {avg_sg_f}')
    plt.axvline(x=avg_pl_m, color='green', linestyle='--', label=f'Average plural masculine: {avg_pl_m}')
    plt.axvline(x=avg_pl_f, color='purple', linestyle='--', label=f'Average plural feminine: {avg_pl_f}')

    plt.legend(loc='upper right')
    plt.title('Visualization of occupation words on gender direction')
    plt.show()


def plot_on_both_directions(pairs, model, semantic_gender_direction, grammatical_gender_direction):
    
    x_coords = []
    y_coords = []
    word_labels = []
    counter = 0
    
    for pair in pairs:
        f_word = pair[0]
        m_word = pair[1]
        x_coords.append(model[f_word].dot(semantic_gender_direction))
        y_coords.append(model[f_word].dot(grammatical_gender_direction))
        word_labels.append(pair[0])
        x_coords.append(model[m_word].dot(semantic_gender_direction))
        y_coords.append(model[m_word].dot(grammatical_gender_direction))
        word_labels.append(pair[1])
        
    # display scatter plot
    fig=plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(x_coords, y_coords, s=10,marker='x',c='black')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        color = 'red' if k%2==0 else 'blue'  # masculine words in blue / feminine words in red
        
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=12,
                     color=color)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Visualization of Spanish nouns on gender directions')
    plt.xlabel('Semantic Gender Direction', fontsize=18)
    plt.ylabel('Grammatical Gender Direction', fontsize=18)
    plt.tick_params(labelsize=10)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

def get_most_biased_words(words, model, direction, topK=500):
    dot_products = {} # es un diccionario?
    
    for i, word in enumerate(words): # nos estamos guardando los índices
        if word not in model: continue 
        dot_products[i] = (model[word]/np.linalg.norm(model[word])).dot(direction/np.linalg.norm(direction))
        
    sorted_dot_products = sorted(dot_products.items(), key=lambda kv: kv[1])
    smallest = sorted_dot_products[0:topK]
    length = len(sorted_dot_products)
    largest = sorted_dot_products[length:(length-topK-1):-1]
    return smallest, largest

def project(emb1, emb2):
    return(emb1.dot(emb2)/np.linalg.norm(emb2))

def get_projections_on_dir(vectors, direction):

    projections = []
    
    for vector in vectors:
        projections.append(project(vector, direction))

    return projections
        
def plot_most_biased(words, model, direction, size, topK=500):
    
    smallest, largest = get_most_biased_words(words, model, direction, topK)
    
    # smallest_50 y largest_50 de la forma [(76, -0.38608473426235956), (592, -0.3807393085065296), ...]
    # donde 76, 592, ... son índices en el array words y -0,38... son las proyecciones en la dirección de género
    
    female_biased = []
    male_biased = []
    
    for i in range(50):
    
        female_entry = [] # Word ID, Spanish word, Cosine similarity
        female_entry.append(smallest[i][0]) # se guardan los índices (Word ID)
        female_entry.append(words[smallest[i][0]]) # Spanish word
        female_entry.append(smallest[i][1]) # Cosine similarity
    
        female_biased.append(female_entry)
    
        male_entry = [] # Word ID, Spanish word, Cosine similarity
        male_entry.append(largest[i][0])
        male_entry.append(words[largest[i][0]])
        male_entry.append(largest[i][1])
    
        male_biased.append(male_entry)

        # female_biased y male_biased de la forma [[76, 'actriz', -0.38608473426235956], [592, 'bailarina', -0.3807393085065296], ...]
        # igual que smallest_50 y largest_50, solo que ahora indicamos que smallest son female biased y largest male biased
        # y también añadimos la palabra

    female_matrix = []
    male_matrix = []
    
    for i in range(len(female_biased)):
        female_matrix.append(model[words[female_biased[i][0]]])
        male_matrix.append(model[words[male_biased[i][0]]])

    # female_matrix y male_matrix tienen los vectores correspondientes a las entries de female_biased y male_biased
    
    female_biased_embeddings = np.array(female_matrix)
    male_biased_embeddings = np.array(male_matrix)

    biased_embeddings = np.concatenate((female_biased_embeddings, male_biased_embeddings), axis=0) # 100 embeddings, 50 female-biased + 50 male-biased

    # queremos reducir a 2 dimensiones biased_embeddings, para visualizarlos en una gráfica
    
    x_coords = get_projections_on_dir(biased_embeddings, direction)
    
    #x_coords = - x_coords # para no liar, que los embeddings femeninos tengan componentes negativas

    pca = PCA(n_components = 10)
    
    fitted = pca.fit_transform(biased_embeddings)
    y_coords = fitted[:,0] # proyecciones en la primera componente principal
    labels = []

    for i in range(len(female_biased)):
        labels.append(words[female_biased[i][0]])

    for i in range(len(male_biased)):
        labels.append(words[male_biased[i][0]])

    fig=plt.figure(figsize=(12, 4), dpi = 80)
    ax1 = fig.add_subplot(111)

    ax1.scatter(x_coords[:50], y_coords[:50], s=10, c='r', marker="o", label='female-biased')
    ax1.scatter(x_coords[50:],y_coords[50:], s=10, c='b', marker="x", label='male-biased')

    print("10 most male biased projections: {} ".format(labels[50:60]))
    print("_________") 
    print("10 most female biased projections: {} ".format(labels[:10]))

    for k, (label, x, y) in enumerate(zip(labels, x_coords, y_coords)):   
        
        if k>=50:
            color = "blue"
        else:
            color = "red"

        if k<10 or (k>49 and k<60):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10, color=color)

    plt.xlim(-size, size)
    plt.ylim(min(y_coords) - 1, max(y_coords) + 3)

    plt.ylabel('PC1', fontsize=10)
    plt.xlabel('Dirección de género semántico', fontsize=10)
        
    plt.legend(loc='upper left');
    plt.show()
    
    most_male_biased = labels[50:70]
    most_female_biased = labels[:20]

    return most_male_biased, most_female_biased



    