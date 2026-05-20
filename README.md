# NLP — Classification & Sentiment des données Twitter

> Projet pédagogique : comprendre comment une machine prend une décision à partir du texte.

---

## Objectif

Explorer le fonctionnement du NLP supervisé à travers la classification de tweets en deux catégories :
- `1` — Positif
- `0` — Négatif

La leçon centrale : **le NLP fonctionne par probabilités, jamais par certitude.**

---

## Dataset

- **Source** : Dataset Twitter labelisé manuellement
- **Taille** : 20 000 tweets
- **Équilibre** : 50% positif / 50% négatif
- **Colonnes** : `targets` (0 ou 1), `text` (tweet brut), `text_clean` (tweet nettoyé)

- Accéder à la base de données: https://www.kaggle.com/datasets/kazanova/sentiment140

---

## Pipeline

```
Données brutes
     │
     ▼
Prétraitement (Text Cleaning)
  - Minuscules
  - Suppression URLs, mentions (@), hashtags (#)
  - Suppression ponctuation et chiffres
  - Tokenisation
  - Suppression stop words
  - Lemmatisation
     │
     ▼
Vectorisation TF-IDF
     │
     ▼

Split Train / Test (80% / 20%, stratifié)
 
     │
     ▼
Modèles de classification
  - Logistic Regression
  - Naive Bayes (MultinomialNB)
     │
     ▼
Évaluation
  - ROC-AUC
  - Accuracy, Precision, Recall, F1 Score
```

---

## Structure du projet

```
├── notebook.ipynb
├── models/
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   └── features.pkl
└── README.md
```

---

## Résultats

| Métrique  | LR Train | LR Test | NB Train | NB Test |
|-----------|----------|---------|----------|---------|
| Accuracy  | 84.6%    | 74.1%   | 87.4%    | 72.4%   |
| Precision | 83.9%    | 73.0%   | 89.6%    | 74.8%   |
| Recall    | 85.6%    | 76.3%   | 84.7%    | 67.6%   |
| F1 Score  | 84.7%    | 74.6%   | 87.1%    | 71.0%   |
| AUC       | —        | 0.82    | —        | 0.81    |

### Conclusion

La **Logistic Regression** est le meilleur modèle sur ce dataset :
- AUC test de **0.82** contre 0.81 pour Naive Bayes
- Écart train/test de **10 points** contre 15 points pour NB (moins d'overfitting)
- Meilleur Recall sur le test (**76.3%** vs 67.6%)

---

## Technologies

| Outil | Usage |
|-------|-------|
| `pandas` | Manipulation du dataset |
| `re` | Nettoyage du texte (regex) |
| `nltk` | Tokenisation, stop words, lemmatisation |
| `scikit-learn` | TF-IDF, modèles, métriques |
| `matplotlib` | Courbes ROC-AUC |
| `joblib` | Sauvegarde des modèles |

---

## Installation

```bash
pip install pandas nltk scikit-learn matplotlib joblib
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## Utilisation des modèles sauvegardés

```python
import joblib

lr_model   = joblib.load('models/lr_model.pkl')
nb_model   = joblib.load('models/nb_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

tweet = ["I love this so much, amazing day !"]
vec   = vectorizer.transform(tweet)

print("LR →", "Positif" if lr_model.predict(vec)[0] == 1 else "Négatif")
print("NB →", "Positif" if nb_model.predict(vec)[0] == 1 else "Négatif")
```

---

## Concepts clés

**Bag of Words (BoW)** — représente un texte par la fréquence de chaque mot. L'ordre est perdu.

**TF-IDF** — pondère chaque mot par sa rareté dans le corpus. Les mots fréquents partout (« ce », « je ») ont un poids faible ; les mots discriminants (« déteste », « adore ») ont un poids fort.

**Naive Bayes** — classifieur probabiliste basé sur le théorème de Bayes. Rapide, simple, efficace sur le texte.

**Logistic Regression** — modèle linéaire qui prédit une probabilité d'appartenance à une classe. Meilleure généralisation ici.

**ROC-AUC** — mesure la capacité du modèle à distinguer les classes. Un AUC de 1.0 est parfait, 0.5 est aléatoire.

---

*Projet réalisé dans le cadre du cours NLP — Classification & Sentiment*
