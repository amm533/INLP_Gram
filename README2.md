# INLP-Gram: Preserving Grammatical Gender when Debiasing Word Embeddings in Spanish 

This repository contains the code and data for the experiments from the paper ["Preserving Grammatical Gender when Debiasing Word Embeddings in Spanish"](url).

To cite:

```



```

## Abstract

Word embeddings are widely used in Natural Language Processing but often encode gender biases, which can lead to discriminatory outcomes. Various debiasing techniques exist, especially focusing on English, thus failing to account for the complexities of languages with grammatical gender, such as Spanish. In this paper, we propose INLP-Gram, an adaptation of the Iterative Nullspace Projection (INLP) algorithm, designed to mitigate gender bias in Spanish word embeddings while preserving grammatical gender information. Our approach refines the INLP method by introducing a mechanism that differentiates between semantic and grammatical gender, ensuring that grammatical gender features are protected across the INLP iterations. We evaluate INLP-Gram using the Word Embedding Association Test (WEAT) and a newly developed grammatical gender classification test. Our results demonstrate that INLP-Gram effectively reduces gender bias while maintaining grammatical gender distinctions. This work advances bias mitigation techniques for word embeddings in linguistically and morphologically-rich languages.

## Algorithm

The implementaion of the INLP-Gram method can be found at the jupyter notebook `notebooks/INLP-Gram.ipynb`. 

`src/debias.py` contains functions from the original INLP algorithm as well as our INLP-Gram functions.


## File Specification

We provide all the code used for this paper in jupyter notebooks and python scripts. The word sets for the experiments are also available.
Spanish embeddings can be downloaded at ([SBWC](https://github.com/dccuchile/spanish-word-embeddings)); and debiased embeddings can be obtained executing our code.

* `data/directions` folder contains the 3 gender directions for Spanish (dPCA: gender direction obtained with PCA, dSem: semantic gender direction, dGram: grammatical gender direction)
* `data/sets_palabras` folder contains the sets of words used in the different experiments (ES: Spanish, FR: French, VAL: Valencian)
  * `data/sets_palabras/ES/Gram/ES_GGCTest_f.txt` and `data/sets_palabras/ES/Gram/ES_GGCTest_m.txt` are the word sets for the grammatical gender classification test
  * `data/sets_palabras/ES/Gram/ES_gram_LDA_f.txt` and `data/sets_palabras/ES/Gram/ES_gram_LDA_m.txt` are the word sets for computing the grammatical gender direction with LDA
* `notebooks/calculate_dGram.ipynb` contains the code for computing the grammatical gender direction
* `notebooks/calculate_dSem.ipynb` contains the code for computing the semantic gender direction
* `notebooks/INLP-Gram.ipynb` and `notebooks/INLP.ipynb` contain the execution for INLP-Gram and INLP algorithms
* `notebooks/WEAT.ipynb`, `notebooks/Gram_Test.ipynb` and `notebooks/generate_plots.ipynb` are used for obtaining the results to our experiments (Sections 5.2.1, 5.2.2 and 5.2.3 in the paper)
* the folder `src` contains python scripts with functions used in the different notebooks
