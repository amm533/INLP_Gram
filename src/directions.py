from gensim.models.keyedvectors import KeyedVectors
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from numpy import linalg as LA
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import preprocessing

############################################################
#        Original code (adapted): Examining Gender Bias in Languages with Grammatical Gender
#        https://github.com/shaoxia57/Bias_in_Gendered_Languages/blob/master/bias_emnlp19.ipynb

def doPCA(pairs, kv_model, num_components = 10):

    matrix = []
    words = []
    
    for a, b in pairs:
        center = ((kv_model[a] + kv_model[b])/2) 
        matrix.append(kv_model[a] - center)
        matrix.append(kv_model[b] - center)

        words.append(a)
        words.append(b)

    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)

    return pca, matrix, words
    

def get_SG_component(v, dGram):
    """
    Get the semantic gender component for a vector by removing the grammatical gender component
    """
    dGram_square_norm = dGram.dot(dGram) # The dot product of a vector with itself equals the square of its magnitude/length/vector norm
    unit_dGram = dGram/ dGram_square_norm # A vector divided by its norm is a unit vector (only has a direction)
    dot_prod = v.dot(dGram) # The dot product is the similarity between vectors
    dGram_component = dot_prod * unit_dGram # we remove this much (amount) in the dGram direction
    SG_component = v - dGram_component
    
    return SG_component


####################################################
#          Other functions

def cos_sim(v1, v2):
    dot_prod = v1.dot(v2)
    cos_sim = dot_prod / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

# the projection of vector a on vector b is equal to the dot product of vector a and vector b, divided by the magnitude of vector b

def project(v1, v2):
    """ Returns the scalar projection of v1 onto v2 """
    dot_prod = v1.dot(v2)
    proj = dot_prod / np.linalg.norm(v2)
    return proj

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def compute_balanced_vector(X1, X2, d_Gram):
    """
    Compute a vector v that satisfies the condition that the sum of dot products
    of vectors in class 1 equals the opposite (negative) of the sum of dot products of vectors in class 2.

    Parameters:
        X1 (numpy.ndarray): Array of shape (N1, d), where N1 is the number of vectors in class 1.
        X2 (numpy.ndarray): Array of shape (N2, d), where N2 is the number of vectors in class 2.
        d_Gram (numpy.ndarray): Initial direction, a vector of shape (d,).

    Returns:
        numpy.ndarray: The computed vector v of shape (d,).
    """
    # Compute the difference vector c
    c = np.sum(X1, axis=0) + np.sum(X2, axis=0)  # Note the addition to achieve opposite sum condition

    # Ensure c is not a zero vector to avoid division by zero
    if np.linalg.norm(c) == 0:
        raise ValueError("The difference vector c is zero, which means the two classes are already balanced.")

    # Project d_Gram orthogonal to c
    v = d_Gram - (np.dot(d_Gram, c) / np.dot(c, c)) * c

    return v
    
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    return math.degrees(angle_rad)


def get_gender_direction_LDA(model, file_f, file_m): 
    """ 
    LDA: Linear Discriminant Analysis.
    LDA seeks to find a linear combination of features (a direction vector) that maximizes the separation between classes while minimizing the scatter within each class. This is achieved by solving a generalized eigenvalue problem.
    :param file_f: file name with feminine words 
    :param file_m: file name with masculine words 
    """

    # Prepare the data (X)
    
    m_words = []
    f_words = []
    
    with open(file_m, "r", encoding="utf-8") as f: 
        for line in f:
            word = line.strip()
    
            if word in m_words:
                continue
                
            if word not in model:
                continue
                    
            m_words.append(word)

    with open(file_f, "r", encoding="utf-8") as f: 
        for line in f:
            word = line.strip()
    
            if word in f_words:
                continue
                
            if word not in model:
                continue        
                
            f_words.append(word)

    grammar_pairs = []
    
    for f,m in zip(f_words, m_words):
        pair = [f,m]
        grammar_pairs.append(pair)

    size = len(grammar_pairs)

    X = np.zeros((2*size, 300))
    
    counter = 0
    
    for pair in grammar_pairs: 
        X[counter] = model[pair[0]]
        counter += 1
        X[counter] = model[pair[1]]
        counter += 1

    # Prepare the labels (y)
    
    y = np.tile([1,2], size)

    # Preprocess the data (standardize)

    # the LDA model assumes that the input dataset has a Gaussian distribution
    
    # In general, many learning algorithms such as linear models benefit from standardization of the data set
    
    # Standardization: standard normally distributed data: Gaussian with zero mean and unit variance.
    print("_______________")
    print("  Standardization")
    print("_______________")
    
    scaler = preprocessing.StandardScaler().fit(X)

    print(f'Original data mean for every dimension: {scaler.mean_[:5]}')
    print(f'Original data variance for every dimension: {scaler.scale_[:5]}')
    print("_______________")
    
    X_scaled = scaler.transform(X)
    print(f'X[0][0] example: {X[0][0]}')
    print(f'X_scaled[0][0] example: {X_scaled[0][0]}')

    scaler = preprocessing.StandardScaler().fit(X_scaled)

    print(f'Standardized data mean for every dimension: {scaler.mean_[:5]}')
    print(f'Standardized data variance for every dimension: {scaler.scale_[:5]}')
    print("Scaled data has zero mean and unit variance")
    print("_______________")

    # Perform parameter tuning for different solvers
    
    solvers = ['svd', 'lsqr', 'eigen']
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  

    for solver in solvers:
        print(f"Evaluating solver: {solver}")
        
        clf_LDA = LinearDiscriminantAnalysis(solver=solver)
        scores = cross_val_score(clf_LDA, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"Scores with solver {solver}: {scores}")
        print(f"Accuracy for solver {solver}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print("---------------")

    clf_LDA = LinearDiscriminantAnalysis(solver='lsqr') # lsqr: Solves least-squares problems directly; often faster for large datasets.
    # When using LDA with n_components = 1, the solution corresponds to the eigenvector associated with the largest eigenvalue 
    #   of the between-class scatter matrix and within-class scatter matrix ratio
    scores = cross_val_score(clf_LDA, X_scaled, y, cv=cv, scoring='accuracy')
    print(f" Grammatical gender direction; LDA Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print("______________________________________________________________________________________________")
    
    clf_LDA.fit(X_scaled, y)
    
    coef = clf_LDA.coef_
    print(f" shape of weights estimated by LDA model: {coef.shape}")
    print(f" Norm of the weight vector: {np.linalg.norm(coef)}")

    # Unit vector, linear combination of features that separates the 2 classes
    direction = np.reshape(coef / np.linalg.norm(coef), (300,))

    proj_class_1 = np.array([project(x, direction) for x in X_scaled[np.where(y == 1)[0]]])  # Feminine
    proj_class_2 = np.array([project(x, direction) for x in X_scaled[np.where(y == 2)[0]]])  # Masculine

    avg_proj_class_1 = np.mean(proj_class_1)
    avg_proj_class_2 = np.mean(proj_class_2)

    print(f"Average scalar projection of feminine words onto dGram: {avg_proj_class_1:.4f}")
    print(f"Average scalar projection of masculine words onto dGram: {avg_proj_class_2:.4f}")
    
    return direction


def get_gender_direction_LDA_emb(gram_emb_set_f, gram_emb_set_m): 
    """ 
    LDA: Linear Discriminant Analysis.
    LDA seeks to find a linear combination of features (a direction vector) that maximizes the separation between classes while minimizing the scatter within each class. This is achieved by solving a generalized eigenvalue problem.
    :param file_f: file name with feminine words 
    :param file_m: file name with masculine words 
    """

    # Prepare the data (X)
    grammar_pairs = []
    
    for f,m in zip(gram_emb_set_f, gram_emb_set_m):
        pair = [f,m]
        grammar_pairs.append(pair)

    size = len(grammar_pairs)

    X = np.zeros((2*size, 300))
    
    counter = 0
    
    for pair in grammar_pairs: 
        X[counter] = pair[0]
        counter += 1
        X[counter] = pair[1]
        counter += 1

    # Prepare the labels (y)
    
    y = np.tile([1,2], size)

    # Preprocess the data (standardize)

    # the LDA model assumes that the input dataset has a Gaussian distribution
    # In general, many learning algorithms such as linear models benefit from standardization of the data set
    
    # Standardization: standard normally distributed data: Gaussian with zero mean and unit variance 
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  
        
    clf_LDA = LinearDiscriminantAnalysis(solver='lsqr') # lsqr: Solves least-squares problems directly; often faster for large datasets.
    scores = cross_val_score(clf_LDA, X_scaled, y, cv=cv, scoring='accuracy')
    print(f" Grammatical gender direction; LDA Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print("______________________________________________________________________________________________")
    
    clf_LDA.fit(X_scaled, y)
    
    coef = clf_LDA.coef_

    # When using LDA with n_components = 1, the solution corresponds to the eigenvector associated with the largest eigenvalue of the between-class scatter matrix and within-class scatter matrix ratio

    # Unit vector, linear combination of features that separates the 2 classes
    direction = np.reshape(coef / np.linalg.norm(coef), (300,))

    # Compute projections onto the LDA direction
    proj_class_1 = np.array([project(x, direction) for x in X_scaled[np.where(y == 1)[0]]])  # Feminine
    proj_class_2 = np.array([project(x, direction) for x in X_scaled[np.where(y == 2)[0]]])  # Masculine

    avg_proj_class_1 = np.mean(proj_class_1)
    avg_proj_class_2 = np.mean(proj_class_2)

    print(f"Average scalar projection of feminine words onto dGram: {avg_proj_class_1:.4f}")
    print(f"Average scalar projection of masculine words onto dGram: {avg_proj_class_2:.4f}")

    # Check and correct bias: make projections symmetric around zero
    bias_shift = (avg_proj_class_1 + avg_proj_class_2) / 2.0  # Center the projections
    if np.abs(bias_shift) > 1e-6:  # Tolerance for bias detection
        print("Bias detected. Correcting direction...")
        direction -= bias_shift * np.mean(X_scaled, axis=0) / np.linalg.norm(direction)
        direction /= np.linalg.norm(direction)  # Renormalize

        # Recompute projections after correction (optional verification)
        proj_class_1_corr = np.array([project(x, direction) for x in X_scaled[np.where(y == 1)[0]]])
        proj_class_2_corr = np.array([project(x, direction) for x in X_scaled[np.where(y == 2)[0]]])
        avg_proj_class_1_corr = np.mean(proj_class_1_corr)
        avg_proj_class_2_corr = np.mean(proj_class_2_corr)
        print(f"Corrected average projection (Class 1 - Feminine): {avg_proj_class_1_corr:.4f}")
        print(f"Corrected average projection (Class 2 - Masculine): {avg_proj_class_2_corr:.4f}")
    
    return direction
