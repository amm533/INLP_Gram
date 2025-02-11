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

`xx_definitional_pairs.json` files contain the gender definitional pairs (female, male) in each language (EN=English, ES=Spanish, FR=French).
