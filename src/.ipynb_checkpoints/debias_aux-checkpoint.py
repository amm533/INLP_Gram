from gensim.models.keyedvectors import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors 
import sklearn 
import numpy as np 
import random



# source: https://github.com/shauli-ravfogel/nullspace_projection

def project_on_gender_subspaces(gender_vector, model: Word2VecKeyedVectors, n = 2500):
    
    # similar_by_vector: Find the top-N most similar words by vector.
    #   parameter topn: Number of top-N similar keys to return. 
    #   What is a "similar key"? --> the method returns a sequence of topn (key, similarity) pairs. 
    
    # (top 2500 similar) positive sense -> masculin, negative-> fem
    group1 = model.similar_by_vector(gender_vector, topn = n, restrict_vocab=None)
    # top 2500 similar to -gender_vector --> 2500 most different to the gender vector (masculine) -> femenine
    group2 = model.similar_by_vector(-gender_vector, topn = n, restrict_vocab=None) #--> similarity with the -gender_direction vector
    
    # all_sims: all vectors of the vocabulary with their similarity
    # all_sims[i][0] contains the words, 
    # all_sims[i][1] represents the similarity, 
    #        it is NOT the component of the vector on the gender direction, i.e. the scalar projection into the gender direction vector
    #        it is a "score" of how male-biased female-biased or neutral words are 
    all_sims = model.similar_by_vector(gender_vector, topn = len(model.vectors), restrict_vocab=None)
    eps = 0.03 # words with a component on the gender direction under 0.03 are considered neutral

    idx = [i for i in range(len(all_sims)) if abs(all_sims[i][1]) < eps]
    
    # visualizando los vectores en castellano, realmente casi todos los valores están por debajo de 0.03
    # prueba: para visualizar realmente cuántos vectores son considerados neutros
    print("Number of neutral words: ")
    print(len(idx))
    print("Total number of words: ")
    print(len(all_sims))
    # resultado: sí que cuadra, valores parecidos al inglés
    
    samp = set(np.random.choice(idx, size = n))
    neut = [s for i,s in enumerate(all_sims) if i in samp]
    return group1, group2, neut

def get_vectors(word_list: list, model: Word2VecKeyedVectors):
    
    vecs = []
    for w in word_list:
        
        vecs.append(model[w])
    
    vecs = np.array(vecs)

    return vecs

def obtain_masc_fem_neut_vecs(gender_direction, model, num_vectors_per_class= 7500):

    gender_unit_vec = gender_direction/np.linalg.norm(gender_direction)

    masc_words_and_scores, fem_words_and_scores, neut_words_and_scores = project_on_gender_subspaces(gender_direction, model, n = num_vectors_per_class)

    masc_words, masc_scores = list(zip(*masc_words_and_scores))
    neut_words, neut_scores = list(zip(*neut_words_and_scores))
    fem_words, fem_scores = list(zip(*fem_words_and_scores))
    masc_vecs, fem_vecs, neut_vecs = get_vectors(masc_words, model), get_vectors(fem_words, model), get_vectors(neut_words, model)

    print("Masculine words")
    print(masc_words)
    print("-------------------------")
    print("Feminine words")
    print(fem_words)
    print("-------------------------")
    print("Neutral words")
    print(neut_words)

    return masc_vecs, fem_vecs, neut_vecs

def train_test_split(masc_vecs, fem_vecs, neut_vecs):
    random.seed(0)
    np.random.seed(0)

    X = np.concatenate((masc_vecs, fem_vecs, neut_vecs), axis = 0)

    y_masc = np.ones(masc_vecs.shape[0], dtype = int)
    y_fem = np.zeros(fem_vecs.shape[0], dtype = int)
    y_neut = -np.ones(neut_vecs.shape[0], dtype = int)
    
    y = np.concatenate((y_masc, y_fem, y_neut))
    
    X_train_dev, X_test, Y_train_dev, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state = 0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train_dev, Y_train_dev, test_size = 0.3, random_state = 0)
    
    print("Train size: {}; Dev size: {}; Test size: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test