import sys
sys.path.append("../src")
import utils

import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from random import shuffle
import statsmodels
from statsmodels.distributions.empirical_distribution import ECDF

def cos_sim(a, b):
    dot_prod = a.dot(b)
    cos_sim = dot_prod / (np.linalg.norm(a) * np.linalg.norm(b))
    #print(f'dot: {cos_sim}')
    return cos_sim

def getTestStatistic(model, X, Y, A, B):
    differenceOfMeans = 0.0
    differenceOfMeans_X = 0.0
    differenceOfMeans_Y = 0.0

    # X to A and B
    for x in X:
        mean_X_A = 0.0
        for a in A:
            mean_X_A = mean_X_A + cos_sim(model[x], model[a])
        
        mean_X_A = mean_X_A / len(A)

        mean_X_B = 0.0
        for b in B:
            mean_X_B = mean_X_B + cos_sim(model[x], model[b])
        
        mean_X_B = mean_X_B / len(B)

        differenceOfMeans_X = differenceOfMeans_X + mean_X_A - mean_X_B

    differenceOfMeans_X = differenceOfMeans_X / len(X)

    # Y to A and B
    for y in Y:
        mean_Y_A = 0.0
        for a in A:
            mean_Y_A = mean_Y_A + cos_sim(model[y], model[a])
        
        mean_Y_A = mean_Y_A / len(A)

        mean_Y_B = 0.0
        for b in B:
            mean_Y_B = mean_Y_B + cos_sim(model[y], model[b])
        
        mean_Y_B = mean_Y_B / len(B)

        differenceOfMeans_Y = differenceOfMeans_Y + mean_Y_A - mean_Y_B

    differenceOfMeans_Y = differenceOfMeans_Y / len(Y)

    differenceOfMeans = abs(differenceOfMeans_X - differenceOfMeans_Y)
    print(f"The test statistic is: {differenceOfMeans}")

    return differenceOfMeans

def nullDistribution(model, X, Y, A, B):

    XY = X + Y
    print("Generating null distribution...")

    A_null_matrix = np.zeros((len(A), len(XY)))
    B_null_matrix = np.zeros((len(B), len(XY)))

    for i, a in enumerate(A):
        for j, xy in enumerate(XY):
            A_null_matrix[i, j] = cos_sim(model[xy], model[a])

    for i, b in enumerate(B):
        for j, xy in enumerate(XY):
            B_null_matrix[i, j] = cos_sim(model[xy], model[b])

    # Assume both concepts have the same size
    set_size = len(XY) // 2
    print(f"Number of permutations: 100 000")
    distribution = np.zeros(100000)

    # Indices for shuffling
    to_shuffle = list(range(len(XY)))

    for iteration in range(100000):
        shuffle(to_shuffle)

        # Calculate means for shuffled data
        mean_similarity_X_A = 0
        mean_similarity_X_B = 0
        mean_similarity_Y_A = 0
        mean_similarity_Y_B = 0

        for i in range(len(A)):
            for j in range(set_size):
                mean_similarity_X_A += A_null_matrix[i, to_shuffle[j]]
            for j in range(set_size):
                mean_similarity_Y_A += A_null_matrix[i, to_shuffle[j + set_size]]

        for i in range(len(B)):
            for j in range(set_size):
                mean_similarity_X_B += B_null_matrix[i, to_shuffle[j]]
            for j in range(set_size):
                mean_similarity_Y_B += B_null_matrix[i, to_shuffle[j + set_size]]

        # Normalize the means
        mean_similarity_X_A /= (len(A) * set_size)
        mean_similarity_X_B /= (len(B) * set_size)
        mean_similarity_Y_A /= (len(A) * set_size)
        mean_similarity_Y_B /= (len(B) * set_size)

        # Compute the difference for this iteration
        distribution[iteration] = (mean_similarity_X_A - mean_similarity_X_B) - (mean_similarity_Y_A - mean_similarity_Y_B)

    return distribution

def getEntireDistribution(model, X, Y, A, B):

    XY = X + Y
    distribution = []
    print("Getting the entire distribution...");

    for xy in XY:
        similarityToA = 0.0 	
        similarityToB = 0.0 			

        for a in A:	
            similarityToA = similarityToA + cos_sim(model[xy], model[a])
					
        similarityToA = similarityToA /len(A)
		
        for b in B:	
            similarityToB = similarityToB + cos_sim(model[xy], model[b])
					
        similarityToB = similarityToB /len(B)
            
        distribution.append(similarityToA - similarityToB)				
	
    return distribution

def calculateCumulativeProbability(sample, value):
	
    cumulative = -100
    sample.sort()
    ecdf = ECDF(sample)
    cumulative = ecdf(value)
    print(f'P(x<{value}): %.3f' % cumulative)
	
    return cumulative

def effectSize(arr, mean):
	
	effect = mean/np.std(arr)		
	return effect

def getPValueAndEffect(model, X, Y, A, B):
		
    test_statistic = getTestStatistic(model, X, Y, A, B) 
    null_distribution = nullDistribution(model, X, Y, A, B)
    entire_distribution = getEntireDistribution(model, X, Y, A, B)
		
    p_value = 1 - calculateCumulativeProbability(null_distribution, test_statistic)
    effect_size = effectSize(entire_distribution, test_statistic)
    print(f" p-value: {p_value}  ---  effect size: {effect_size}")
    
    return [p_value, effect_size]	

if __name__ == "__main__":
    EN_biased_model = KeyedVectors.load_word2vec_format(fname = "../data/embeddings/vecs.filtered.txt", binary=False)
    ES_biased_model = KeyedVectors.load('../data/embeddings/keyedvectors/model_esp.kv', mmap='r')
    ES_inlp_debiased_model = KeyedVectors.load('../data/embeddings/keyedvectors/ES_inlp_debiased_model.kv', mmap='r')
    inlpES_debiased_model = KeyedVectors.load('../data/embeddings/keyedvectors/inlpES_debiased_model.kv', mmap='r')
    
    # Experiment career-family
    EN_career = ["executive" , "management" , "professional" , "corporation" , "salary" , "office", "business" , "career"]
    EN_career = utils.check_vocabulary(EN_biased_model, EN_career)
    
    EN_family = ["home" , "parents" , "children" , "family" , "cousins" , "marriage" , "wedding" , "relatives"]
    EN_family = utils.check_vocabulary(EN_biased_model, EN_family)
    
    EN_names_m = ["john" , "paul" , "mike" , "kevin" , "steve" , "greg" , "jeff" , "bill"]
    EN_names_m = utils.check_vocabulary(EN_biased_model, EN_names_m)
    
    EN_names_f = ["amy" , "joan" , "lisa" , "sarah" , "diana" , "kate" , "ann" , "donna"]
    EN_names_f = utils.check_vocabulary(EN_biased_model, EN_names_f)
    
    
    ES_career = ["ejecutivo" , "gestión" , "profesional" , "corporación" , "salario" , "oficina", "negocio" , "carrera"]
    ES_career = utils.check_vocabulary(ES_biased_model, ES_career)
    
    ES_family = ["hogar" , "padres" , "niños" , "familia" , "primos" , "boda" , "matrimonio" , "parientes"]
    ES_family = utils.check_vocabulary(ES_biased_model, ES_family)
    
    ES_names_m = ["juan" , "pablo" , "miguel" , "diego" , "marcos" , "hugo" , "sergio" , "javier"]
    ES_names_m = utils.check_vocabulary(ES_biased_model, ES_names_m)
    
    ES_names_f = ["maría" , "julia" , "valeria" , "sofía" , "lucía" , "paula" , "ana" , "isabel"]
    ES_names_f = utils.check_vocabulary(ES_biased_model, ES_names_f)
    
    # Experiment science-arts
    EN_arts = ["poetry" , "art" , "sculpture" , "dance" , "literature" , "novel" , "symphony" , "drama"]
    EN_arts = utils.check_vocabulary(EN_biased_model, EN_arts)
    
    EN_science = ["science" , "technology" , "physics"  , "chemistry" , "hypothesis" , "atom" , "experiment" , "astronomy"]
    EN_science = utils.check_vocabulary(EN_biased_model, EN_science)
    
    EN_m_terms = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    EN_m_terms = utils.check_vocabulary(EN_biased_model, EN_m_terms)
    
    EN_f_terms =["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    EN_f_terms = utils.check_vocabulary(EN_biased_model, EN_f_terms)
    
    ES_arts = ["poesía", "arte", "escultura", "danza", "literatura", "novela", "sinfonía", "drama"]
    ES_arts  = utils.check_vocabulary(ES_biased_model, ES_arts )
    
    ES_science = ["ciencia", "tecnología", "física", "química", "hipótesis", "átomo", "experimento", "astronomía"]
    ES_science = utils.check_vocabulary(ES_biased_model, ES_science)
    
    ES_m_terms = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tío"]
    ES_m_terms = utils.check_vocabulary(ES_biased_model, ES_m_terms)
    
    ES_f_terms =["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]
    ES_f_terms = utils.check_vocabulary(ES_biased_model, ES_f_terms)
    
    # Experiment math-arts
    EN_arts = ["poetry" , "art" , "sculpture" , "dance" , "literature" , "novel" , "symphony" , "drama"]
    EN_arts = utils.check_vocabulary(EN_biased_model, EN_arts)
    
    EN_math = ["math" , "algebra" , "geometry" , "calculus" , "equations" , "computation" , "numbers" , "addition"]
    EN_math = utils.check_vocabulary(EN_biased_model, EN_arts)
    
    EN_m_terms = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    EN_m_terms = utils.check_vocabulary(EN_biased_model, EN_m_terms)
    
    EN_f_terms =["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
    EN_f_terms = utils.check_vocabulary(EN_biased_model, EN_f_terms)
    
    ES_arts = ["poesía", "arte", "escultura", "danza", "literatura", "novela", "sinfonía", "drama"]
    ES_arts  = utils.check_vocabulary(ES_biased_model, ES_arts )
    
    ES_math = ["matemáticas", "álgebra", "geometría", "cálculo", "ecuaciones", "computación", "números", "suma"]
    ES_math  = utils.check_vocabulary(ES_biased_model, ES_math)
    
    ES_m_terms = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tío"]
    ES_m_terms = utils.check_vocabulary(ES_biased_model, ES_m_terms)
    
    ES_f_terms =["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]
    ES_f_terms = utils.check_vocabulary(ES_biased_model, ES_f_terms)

    # Test execution
    # English - original embeddings
    print("English - original embeddings")
    getPValueAndEffect(EN_biased_model, EN_names_m, EN_names_f, EN_career, EN_family)
    getPValueAndEffect(EN_biased_model, EN_science, EN_arts, EN_m_terms, EN_f_terms)
    getPValueAndEffect(EN_biased_model, EN_math, EN_arts, EN_m_terms, EN_f_terms)

    # Spanish - original embeddings
    print("Spanish - original embeddings")
    getPValueAndEffect(ES_biased_model, ES_names_m, ES_names_f, ES_career, ES_family)
    getPValueAndEffect(ES_biased_model, ES_science, ES_arts, ES_m_terms, ES_f_terms)
    getPValueAndEffect(ES_biased_model, ES_math, ES_arts, ES_m_terms, ES_f_terms)

