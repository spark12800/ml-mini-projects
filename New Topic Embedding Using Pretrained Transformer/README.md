# News Topic Clustering with Transformer Embeddings

**Goal:**  
Evaluate how well pretrained Transformer embeddings separate news topics in the **AG News** dataset.

---

## Summary

This project compares sentence embeddings from:

- **all-MiniLM-L6-v2**
- **BERT-base-uncased**
- **RoBERTa-base**

Headlines are encoded via **mean pooling** of last-hidden-state representations and **L2-normalised** for stable probing.  
Topic structure is analysed using:

- **PCA → t-SNE** for 2D visualisation  
- **Linear probes** (Logistic Regression, Linear SVM, Ridge)

---

## Dataset

This project uses the **AG News** dataset from the *HuggingFace Datasets* library:

- **Source:** Zhang, Zhao & LeCun (2015), "Character-level Convolutional Networks for Text Classification"
- **Access:** `datasets.load_dataset("ag_news")`
- **Structure:** 4 balanced topic classes — *World*, *Sports*, *Business*, *Sci/Tech*
- **Size:** 120,000 training samples and 7,600 test samples (headlines + labels)

---

## Method

**Embedding Extraction**
- Tokenise (max_len=128) → model forward pass  
- Mean pooling with attention mask  
- L2-normalised sentence vectors  

**Linear Probing**
- Train 3 linear classifiers (e.g., Multinomial Logistic Regression, Linear SVM, Ridge Classifier)
- Select best model by **F1-macro** across 4 topics  
  (*World, Sports, Business, Sci/Tech*)

**Visualisation**
- PCA (50 dims) → t-SNE (2 dims)  
- Per-topic colour-coded clusters

---

## Key Findings

- **MiniLM-L6-v2** produced the most separable topic clusters and strongest probe performance.  
- **RoBERTa-base** competitive but slightly less linearly separable.  
- **BERT-base** showed weaker clustering in both probes and t-SNE.  

---

**Tools:** HuggingFace Transformers, SentenceTransformers, scikit-learn  
**Notebook:** `News_Topic_Clustering.ipynb`
