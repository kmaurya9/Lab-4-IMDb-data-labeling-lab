# Lab 4 — Data Labeling: IMDb Review Sentiment

**Author:** Kshitij Maurya
**Course:** MLOps (IE7305) — Northeastern University

This lab labels IMDb movie reviews as **positive** or **negative** without hand-labeling, using **weak supervision** (rule-based labeling functions), balances the labels with **text augmentation**, and trains classifiers that are evaluated **overall and on data slices** against the true labels.

---

## Pipeline

```
data/imdb_reviews.csv (8,000 reviews)
        │
        ▼
01_Data_Labeling.ipynb      7 labeling functions → label matrix → majority vote
        │                   (20% held out with true labels as the test set)
        ▼
02_Data_Augmentation.ipynb  synonym replacement / random swap / random deletion
        │                   (negation words protected) → balanced classes
        ▼
03_Model_Development.ipynb  TF-IDF + numeric features → LogReg / Naive Bayes / Linear SVM
                            5-fold CV → threshold → test on true labels → slice analysis
```

Because IMDb includes true labels, the lab can measure how accurate the weak labels are, and how well a model trained only on them works on real labels.

---

## Project structure

```
Lab4/
├── notebooks/
│   ├── 01_Data_Labeling.ipynb       # labeling functions + majority vote
│   ├── 02_Data_Augmentation.ipynb   # label-safe text augmentation
│   └── 03_Model_Development.ipynb   # training, evaluation, slicing
├── src/
│   ├── utils.py                     # loading/saving, text cleaning, rating extraction
│   ├── config.py                    # reads config.yaml
│   └── data_validator.py            # data-quality checks
├── tests/                           # pytest unit tests for src/
├── data/                            # imdb_reviews.csv (+ generated CSVs)
├── outputs/                         # model, predictions, figures (generated)
├── config.yaml                      # paths, keyword lists, model settings
├── Makefile
└── requirements.txt
```

CI: `.github/workflows/ci.yml` runs the lint check and unit tests on every push to `main`.

---

## How to run

```bash
git clone https://github.com/kmaurya9/Lab-4-IMDb-data-labeling-lab.git
cd Lab-4-IMDb-data-labeling-lab
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                       # dependencies + NLTK WordNet

make run                           # executes notebooks 01 → 02 → 03
# or open them yourself:  jupyter notebook notebooks/

make test                          # unit tests
```

The dataset (`data/imdb_reviews.csv`) is included, so no download is needed. Runtime is about 2 minutes on a laptop.

---

## Results

| Step | Result |
|------|--------|
| Labeling-function coverage | 52.5% of training reviews got at least one vote (4.9% conflicting) |
| Weak-label accuracy (vs. true labels) | 88.8% on the 3,114 labeled reviews |
| Augmentation | 3,114 → 3,490 reviews, classes balanced 50 / 50 |
| Best model (5-fold CV on weak labels) | Linear SVM, ROC-AUC 0.974 |
| **Test accuracy (true labels)** | **0.836** |
| **Test ROC-AUC (true labels)** | **0.925** |
| Weakest slice | short reviews (< 100 words), accuracy 0.819 |

The rules label only about half the reviews, but the model trained on them reaches 83.6% accuracy on **all** test reviews. It learns words and phrases beyond the hand-written keywords.

---

## Tech stack

Python 3.10+, pandas, NumPy, scikit-learn, NLTK (WordNet), matplotlib, seaborn, PyYAML, joblib, pytest, GitHub Actions.
