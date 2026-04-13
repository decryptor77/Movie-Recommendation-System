# Movie Recommendation System Challenge

A hybrid recommendation system built on a curated sample of the MovieLens 32M dataset for predicting user movie ratings.  
The project combines **implicit-feedback pretraining**, **Neural Matrix Factorization (NeuMF)**, **content features**, and **multi-seed ensembling** to improve rating prediction under the competition’s **Weighted RMSE (WRMSE)** metric. 
<br>Done during my Data Science & Machine Learning MSc at Reichman University.

## Result

- **Final WRMSE:** `0.797`
- **Competition outcome:** **Top-5 winning result**
- **Framework:** PyTorch + CUDA
- **Hardware used during experimentation:** Colab GPU environments, including H100/A100

## Problem

The task is to predict unseen **explicit movie ratings** on a scale of **0.5 to 5.0**.  
The dataset also includes **implicit feedback**: missing ratings indicate that a user interacted with a movie but did not explicitly rate it.

Submissions are evaluated using **Weighted RMSE (WRMSE)**:

$$
\mathrm{WRMSE}=\sqrt{\frac{\sum_i w_i(\hat{y}_i-y_i)^2}{\sum_i w_i}}
$$

where

$$
w_i=\frac{1}{\sqrt{\text{number of ratings for the movie}}}
$$

This reduces the dominance of very popular movies and makes robust generalization across the catalog more important.

## Approach

The final system uses a **two-stage hybrid recommendation pipeline**:

### 1. Implicit-feedback pretraining
A lightweight dot-product model is pretrained with **BPR (Bayesian Personalized Ranking)** on implicit interactions to learn strong user/item representations from “watched” behavior.

### 2. Explicit-rating prediction
The pretrained embeddings are used to initialize a **NeuMF-style model** that combines:

- **GMF branch** for linear collaborative interactions
- **MLP branch** for nonlinear user-item interactions
- **User/item bias terms**
- **Content features** from:
  - movie titles
  - genres
  - user-defined tags

The rating model is optimized with a **WRMSE-aligned weighted MSE loss**.

### 3. Ensembling
The final submission is produced with a **multi-seed weighted ensemble**, which reduces variance and improves generalization.

## Key Features

- Hybrid recommendation model: **BPR pretraining + NeuMF + content features**
- Uses both **implicit and explicit feedback**
- **Metadata preprocessing**:
  - ID remapping
  - title cleaning
  - metadata normalization
  - tag deduplication / normalization
  - content tokenization
- **User-wise holdout validation**
- Offline evaluation with:
  - **WRMSE**
  - **HR@K**
  - **NDCG@K**
- **Cold-start handling** for unseen items using content features + fallback priors
- GPU-accelerated training with **PyTorch + CUDA**
- Efficient implementation with vectorized operations and batch inference

## Repository Structure

```text
.
├── README.md
├── movielens_recommender.ipynb
├── train.csv
├── movies.csv
├── tags.csv
└── ratings_submission.csv
```

---

## Author

**Noor Nashef**  
MSc Data Science & Machine Learning student, Reichman University <br>
BSc in Information Systems Engineering specialized in Machine Learning, Technion
