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



def classify_grammatical_gender(pares, model):

    f_vecs = []
    m_vecs = []

    n_f = 0
    n_m = 0

    for par in pares:
        if par[0] in model:
            f_vecs.append(model[par[0]])
            n_f = n_f + 1
        
        if par[1] in model:    
            m_vecs.append(model[par[1]])
            n_m = n_m + 1
    
    X = f_vecs + m_vecs

    labels = [0] * n_f + [1] * n_m

    cv = StratifiedKFold(n_splits=1, shuffle=True, random_state=42) # preserve the percentage of samples for each class
    
    print("K-NN")
    print("_________")   

    # Parameter tuning: best k (number of nearest-neighbors)

    k_values = [i for i in range (1,31)]
    scores = []
    max_score = 0
    max_n = -1

    for k in k_values:
        
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_score = cross_val_score(knn, X, labels, cv=cv, scoring='accuracy')
        scores.append(np.mean(cv_score))
        
        if np.mean(cv_score) > max_score:
            max_score = np.mean(cv_score)
            max_n = k

    knn_score = max_score
    
    plt.plot(k_values, scores, marker='o')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.title("KNN Accuracy Scores for Different K Values")
    plt.show()

    print(f"Accuracy: {100*knn_score}%")

    print()
    print("Logistic Regression")
    print("_________")   

    # No parameter tuning
    
    logreg = LogisticRegression(random_state=16)

    cv_score = cross_val_score(logreg, X, labels, cv=cv, scoring='accuracy')
    logreg_score = np.mean(cv_score)

    print(f"Accuracy: {100*logreg_score}%")

    print()
    print("Support Vector Classification (SVC)")
    print("_________") 

    # Parameter tuning: C (regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors)

    C_values = np.logspace(-3, 3, 10)
    
    mean_scores = []
    
    for C in C_values:
        svm = SVC(C=C, kernel='linear', random_state=42)
        cv_score = cross_val_score(svm, X, labels, cv=cv, scoring='accuracy')
        mean_scores.append(cv_score.mean())

    plt.figure(figsize=(8, 6))
    plt.semilogx(C_values, mean_scores, marker='o', linestyle='--', color='b')
    plt.title('Cross-Validation Accuracy for Different C Values')
    plt.xlabel('C (log scale)')
    plt.ylabel('Mean Cross-Validation Accuracy')
    plt.grid(True)
    plt.show()

    best_C = C_values[np.argmax(mean_scores)]
    svc_score = max(mean_scores)
    
    print(f"Best C value: {best_C}, Accuracy: {100*svc_score}%")

    avg = (knn_score + logreg_score + svc_score) / 3

    print("_________") 
    print(f"Average accuracy: {100*avg}%")

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


def WEAT(model, words, weat_m, weat_f):
    
    # WEAT original: mide si las palabras están más asociadas a un género que al otro
    # (de media, el grado de asociación a 1 solo género)
    
    total_association_tg = 0.0
    
    for word in words:

        if word not in model:
            continue # ignore the pair

        tg_m = pair[0]
        tg_f = pair[1]
        
        sum_cos_sim_att_m = 0.0
        sum_cos_sim_att_f = 0.0
        
        for att_m in weat_m: 
            sum_cos_sim_att_m += directions.cos_sim(model[att_m], model[word]) 
            
        for att_f in weat_f:
            sum_cos_sim_att_f += directions.cos_sim(model[att_f], model[word]) 

        avg_cos_sim_att_m = sum_cos_sim_att_m / len(weat_m)
        avg_cos_sim_att_f = sum_cos_sim_att_f / len(weat_f) 

        association_tg = abs(avg_cos_sim_att_m - avg_cos_sim_att_f) 

        total_association_tg += association_tg 
    
    print("total_association_tg: {}".format(total_association_tg))
        
    return(total_association_tg)
    
def modified_WEAT(model, pairs, weat_m, weat_f):

    total_association_tg_m = 0.0
    total_association_tg_f = 0.0
    
    for pair in pairs:

        if pair[0] not in model or pair[1] not in model:
            continue # ignore the pair

        tg_m = pair[0]
        tg_f = pair[1]
        
        sum_cos_sim_att_m_tg_m = 0.0
        sum_cos_sim_att_m_tg_f = 0.0
        sum_cos_sim_att_f_tg_m = 0.0
        sum_cos_sim_att_f_tg_f = 0.0
        
        for att_m in weat_m: 
            sum_cos_sim_att_m_tg_m += directions.cos_sim(model[att_m], model[tg_m]) # bias -> +
            sum_cos_sim_att_m_tg_f += directions.cos_sim(model[att_m], model[tg_f]) # bias -> -
            
        for att_f in weat_f:
            sum_cos_sim_att_f_tg_m += directions.cos_sim(model[att_f], model[tg_m]) # bias -> -
            sum_cos_sim_att_f_tg_f += directions.cos_sim(model[att_f], model[tg_f]) # bias -> +

        avg_cos_sim_att_m_tg_m = sum_cos_sim_att_m_tg_m / len(weat_m) # bias -> +
        avg_cos_sim_att_m_tg_f = sum_cos_sim_att_m_tg_f / len(weat_m) # bias -> -
        avg_cos_sim_att_f_tg_m = sum_cos_sim_att_f_tg_m / len(weat_f) # bias -> -
        avg_cos_sim_att_f_tg_f = sum_cos_sim_att_f_tg_f / len(weat_f) # bias -> +

        association_tg_m = avg_cos_sim_att_m_tg_m - avg_cos_sim_att_f_tg_m # bias -> +
        association_tg_f = avg_cos_sim_att_m_tg_f - avg_cos_sim_att_f_tg_f # bias -> -

        total_association_tg_m += association_tg_m # bias -> +
        total_association_tg_f += association_tg_f # bias -> +
        
    avg_association_tg_m = total_association_tg_m / len(pairs)
    avg_association_tg_f = total_association_tg_f / len(pairs)
    
    #print("total_female_association: {}".format(total_association_tg_f))
    #print("total_male_association: {}".format(total_association_tg_m))
    #print("male - female: {}".format(total_association_tg_m - total_association_tg_f))
    #print("avg_association_tg_m: {}".format(avg_association_tg_m))
    #print("avg_association_tg_f: {}".format(avg_association_tg_f))
        
    return  [total_association_tg_m - total_association_tg_f, abs(abs(total_association_tg_m)-abs(total_association_tg_f))] 

def gender_association_score(model, w, G):
    """
    Compute the gender association score for a word with respect to 1 gender. 
    Formula: s is the association score for a word
    s(w, G) = (1 / |G|) * Σ (cos(w, g)) for all g ∈ G
    :param model: 
    :param w: gendered word
    :param G: set of gender attribute words, representing 1 gender (ex: 'woman', 'she')
    :return: 
    """
    # ----------------------------------------------------
    # s(w, G) = (1 / |G|) * Σ (cos(w, g)) for all g ∈ G
    # ----------------------------------------------------
    
    if w not in model:
         return 0.0

    s =  []
    for g in G: 
                
        if g not in model:
            continue
                    
        s.append(abs(directions.cos_sim(model[w], model[g])))

    return np.mean(s)

def MWEAT(model, pairs, A, B):
    """
    Modified Word Embedding Association Test (Zhou et al.) 
    Formula: test_stat is the bias for a set of word pairs
    test_stat = ||Σ s(w_m, A, B) for all W_M|| - || Σ s(w_f, A, B) for all W_F|| ;
    s(w, A, B) = (1 / |A|) * Σ (cos(w, a)) for all a ∈ A
               - (1 / |B|) * Σ (cos(w, b)) for all b ∈ B
    :param model: 
    :param pairs: (w_m: target masculine word, w_f: target feminine word)
    :param A: gender 1 attribute words
    :param B: gender 2 attribute words
    :return: 
    """

    all_s_w_f = []
    all_s_w_m = []
    
    for pair in pairs:
        if pair[0] not in model or pair[1] not in model:
            continue # ignore the pair

        w_m = pair[0]
        w_f = pair[1]

        # ----------------------------------------------------
        # s(w, A, B) = (1 / |A|) * Σ (cos(w, a)) for all a ∈ A
        #            - (1 / |B|) * Σ (cos(w, b)) for all b ∈ B
        # ----------------------------------------------------
        
        # s(w_m, A, B)

        # (1 / |A|) * Σ (cos(w_m, a)) for all a ∈ A
        temp = []
        for a in A: 
            temp.append(abs(directions.cos_sim(model[w_m], model[a])))
            
        s_cos_w_m_A = np.mean(temp)   

        # (1 / |B|) * Σ (cos(w_m, b)) for all b ∈ B
        temp = []
        for b in B: 
            temp.append(abs(directions.cos_sim(model[w_m], model[b])))
            
        s_cos_w_m_B = np.mean(temp)  

        s_w_m = s_cos_w_m_A - s_cos_w_m_B
        all_s_w_m.append(s_w_m)

        # ----------------------------------------------------
        # s(w_f, A, B)

        # (1 / |A|) * Σ (cos(w_f, a)) for all a ∈ A
        temp = []
        for a in A: 
            temp.append(abs(directions.cos_sim(model[w_f], model[a])))
            
        s_cos_w_f_A = np.mean(temp)   

        # (1 / |B|) * Σ (cos(w_f, b)) for all b ∈ B
        temp = []
        for b in B: 
            temp.append(abs(directions.cos_sim(model[w_f], model[b])))
            
        s_cos_w_f_B = np.mean(temp)  

        s_w_f = s_cos_w_f_A - s_cos_w_f_B
        all_s_w_f.append(s_w_f)

    # -----------------------------------------------------------------
    # test_stat = ||Σ s(w_m, A, B) for all w|| - || Σ s(w_f, A, B) for all w||

    test_stat = abs(abs(sum(all_s_w_m)) - abs(sum(all_s_w_f)))
        
    return  test_stat

def MWEAT_pval(model, gender_pairs, weat_m, weat_f, k):
    
    test_stat = MWEAT(model, gender_pairs, weat_m, weat_f)
    
    print("MWEAT Diff: {0}".format(test_stat)) 
    
    observed_over = [] 

    all_targets = []
    
    for pair in gender_pairs:
        all_targets.append(pair[0])
        all_targets.append(pair[1])
   
    random.seed(521)
        
    for i in range(k):
        c = random.sample(all_targets, len(gender_pairs)) 
        not_c = list(filterfalse(lambda x: x in c, all_targets)) 
        
        stat = MWEAT(model, list(zip(c, not_c)), weat_m, weat_f)
            
        observed_over.append(stat >= test_stat)
            
        if len(observed_over) % 100 == 0:
            logging.info("  Iteration {0}. Iteration Score: {1}. Mean p-value: {2}".format(len(observed_over), stat, np.mean(observed_over)))

    p_value = np.mean(observed_over)
    
    print(f"Test Statistic p-value: {p_value} (significance level: 0.05)")
    
    return p_value

def MWEAT_pval_old(model, gender_pairs, aw1, aw2, modified=1):
    
    test_stat = modified_WEAT(model, gender_pairs, aw1, aw2)[modified]
    
    print("MWEAT Diff: {0}".format(test_stat)) 
    
    observed_over = [] 

    all_targets = []
    
    for pair in gender_pairs:
        all_targets.append(pair[0])
        all_targets.append(pair[1])

    print(all_targets[:10])    
    random.seed(521)
        
    for i in range(1000):
        c = random.sample(all_targets, len(gender_pairs)) 
        not_c = list(filterfalse(lambda x: x in c, all_targets)) 
        
        stat = modified_WEAT(model, list(zip(c, not_c)), aw1, aw2)[modified] 
            
        observed_over.append(stat > test_stat)
            
        if len(observed_over) % 100 == 0:
            print(len(c))
            print(len(not_c))
            print(len(list(zip(c, not_c))))
            logging.info("  Iteration {0}. Iteration Score: {1}. Mean p-value: {2}".format(len(observed_over), stat, np.mean(observed_over)))
                
    return np.mean(observed_over)

def Gender_Component_Test(gender_pairs, model, gender_direction):
    """
    Returns a score for the gender bias, computed as the deviation from neutrality of the average component in the gender direction of masculina and feminine pairs.
    :param gender_pairs: 
    :param model: 
    :param gender_direction: The gender direction aimed by the debias method.
    (The debias method has the objective of mitigating the bias that is initially observed in this direction.) 
    The component of embeddings in this direction is considered the gender component.
    """

    total_m = 0 # Sum of masculine gender components
    total_f = 0 # Sum of feminine gender components

    count = 0
    
    for pair in gender_pairs:

        if pair[0] not in model or pair[1] not in model:
            continue # ignore the pair

        count = count + 1
        
        # Projections
        component_f = model[pair[0]].dot(gender_direction)
        component_m = model[pair[1]].dot(gender_direction)
        
        total_m = total_m + component_m
        total_f = total_f + component_f

    avg_m = total_m/count
    avg_f = total_f/count
    
    total_avg = (avg_m + avg_f)/2

    return total_avg

def Gender_Component_Test_pval(gender_pairs, model, gender_direction):
    """
    Returns the p-value for the Gender Component Test score: the probability of obtaining a result equal to or more extreme than that observed, i.e. the chance that a type I error was made). If the p-value is less than the significance level of 0.05 then the alternative hypothesis is accepted (the hypothesis that the embeddings are biased)
    :param gender_pairs: 
    :param model: 
    :param gender_direction: The gender direction aimed by the debias method.
    (The debias method has the objective of mitigating the bias that is initially observed in this direction.) 
    The component of embeddings in this direction is considered the gender component.
    """
    
    test_stat = Gender_Component_Test(gender_pairs, model, gender_direction)
    
    print("Gender Component Test Score: {0}".format(test_stat))  
    
    observed_over = [] 

    all_words = []

    for pair in gender_pairs:
        all_words.append(pair[0])
        all_words.append(pair[1])

    random.seed(520)
        
    for i in range(1000):
        
        c = random.sample(all_words, len(gender_pairs)) 
        not_c = filterfalse(lambda x: x in c, all_words)
        stat = Gender_Component_Test(list(zip(c, not_c)), model, gender_direction)    
        
        observed_over.append(abs(stat) >= abs(test_stat)) # "The result is equal to or more extreme (different from 0) than that observed"
            
        if len(observed_over) % 100 == 0:
            logging.info("  Iteration {0}. Iteration Score: {1}. Mean p-value: {2}".format(len(observed_over), stat, np.mean(observed_over)))
                
    return np.mean(observed_over)

def WEAT_s(model, w, A, B):
    """
    Word Embedding Association Test (Calistan et al.) s measure.
    s measures the association of a target word w with the attribute (e.g. gender).
    The higher the value, the stronger is the bias of the word towards one side of the attribute.
    Formula: 
    s(w, A, B) = (1 / |A|) * Σ (cos(w, a)) for all a ∈ A
               - (1 / |B|) * Σ (cos(w, b)) for all b ∈ B
    :param model: 
    :param w: target word
    :param A: set A of attribute words
    :param B: set B of attribute words
    :return: s measure of WEAT
    """

    if w not in model:
         return 0.0

    s_a =  []
    for a in A: 
                
        if a not in model:
            continue
                    
        s_a.append(abs(directions.cos_sim(model[w], model[a])))

    s_a = np.mean(s_a)

    s_b =  []
    for b in B: 
                
        if b not in model:
            continue
                    
        s_b.append(abs(directions.cos_sim(model[w], model[b])))

    s_b = np.mean(s_b)

    s = abs(s_a - s_b)

    return s

def WEAT_effect_size(model, X, Y, A, B, print_val=True):
    """
    Word Embedding Association Test (Calistan et al.) effect size.
    The effect size represents the practical/substantive significance of the difference between the distributions of s measures for the target sets.
    The calculation is made similar to cohen's d, but dividing by the standard deviation of all s instead of the pooled standard deviation. Conventional small, medium, and large values of d are 0.2, 0.5,and 0.8.
    Formula: 
    effect_size = (mean_x∈X[s(x, A, B)] - mean_y∈Y[s(y, A, B)]) / std_dev_w∈(X ∪ Y)[s(w, A, B)]
    :param model: 
    :param X: set X of target words
    :param Y: set Y of target words
    :param A: set A of attribute words
    :param B: set B of attribute words
    :return: effect size of WEAT
    """

    s_x = [WEAT_s(model, x, A, B) for x in X]
    s_y = [WEAT_s(model, y, A, B) for y in Y]

    cohens_d = abs(np.mean(s_x) - np.mean(s_y)) / (math.sqrt((np.std(s_x) ** 2 + np.std(s_y) ** 2) / 2))
    all_s = s_x + s_y
    effect_size = abs(np.mean(s_x) - np.mean(s_y)) / np.std(all_s)

    if(print_val):
        print(f" Cohen's d: {cohens_d}, WEAT effect size: {effect_size}")

    return effect_size

def WEAT_test_statistic(model, X, Y, A, B):
    """
    Word Embedding Association Test (Calistan et al.) test statistic.
    Formula: s(X, Y, A, B) = Σ x∈X[s(x, A, B)] - Σ y∈Y[s(y, A, B)]
    :param model: 
    :param X: set X of target words
    :param Y: set Y of target words
    :param A: set A of attribute words
    :param B: set B of attribute words
    :return: test statistic of WEAT
    """

    s_x = [WEAT_s(model, x, A, B) for x in X]
    s_y = [WEAT_s(model, y, A, B) for y in Y]

    sum_x = sum(s_x)
    sum_y = sum(s_y)

    test_stat = abs(sum_x - sum_y)
    return test_stat
    

def WEAT_p_value(model, X, Y, A, B, max_iterations=1000):
    """
    Word Embedding Association Test (Calistan et al.) p-value.
    The p-value represents the statistical significance of the difference between the distributions of s measures for the target sets.
    The significance threshold used is 1e-2 (equivalent to 0.01). 
    The one-sided P value of the permutation test is computed as:
    p-value = Probability[ s(Xi, Yi, A, B) > s(X, Y, A, B) ] for all iterations
    :param model: 
    :param X: set X of target words
    :param Y: set Y of target words
    :param A: set A of attribute words
    :param B: set B of attribute words
    :return: effect size of WEAT
    """
    test_stat = WEAT_test_statistic(model, X, Y, A, B)
    
    print(" WEAT Test statistic: {0}".format(test_stat)) 
    
    observed_over = [] 

    all_targets = X + Y
   
    for i in range(max_iterations):
        
        c = random.sample(all_targets, len(X))    
        not_c = list(filterfalse(lambda x: x in c, all_targets))        
        stat = WEAT_test_statistic(model, c, not_c, A, B)
            
        observed_over.append(stat > test_stat)
            
        if i % 100 == 0:
            logging.info("  Iteration {0}. Iteration Score: {1}. Mean p-value: {2}".format(i, stat, np.mean(observed_over)))

    p_value = np.mean(observed_over)
    
    print(f"Test Statistic p-value: {p_value} (Significance threshold: 0.01)")
    
    return p_value

    

    