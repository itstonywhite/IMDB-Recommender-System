# 🎬 IMDB Movie Recommender Engine

This repository contains a professional end-to-end Machine Learning pipeline that builds a Recommendation System using the MovieLens dataset. It explores multiple architectural approaches to personalization.

## ⚙️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-0073B7?style=for-the-badge&logo=&logoColor=white)

**📥 Install required packages:**

```text
pip install numpy pandas scipy matplotlib seaborn scikit-learn joblib
```

## 🛠️ Architecture & Algorithms

The project implements and evaluates three distinct recommender models:

1. **Content-Based Filtering:** Utilizes **TF-IDF Vectorization** and Cosine Similarity on movie genres to solve the cold-start item problem.
2. **User-Based Collaborative Filtering:** Leverages **K-Nearest Neighbors (KNN)** to cluster users with similar behaviors, optimizing serendipity in recommendations.
3. **Item-Based Collaborative Filtering:** Uses KNN on a Sparse Matrix (SciPy) to map stable item-to-item relationships based on rating patterns.

## 🚀 Key Takeaways & Export

Following industry standards, the **Item-Based Collaborative Filtering** model was selected for export due to its scalability and computational stability in production environments. The model and its underlying pivot structures are serialized as `.pkl` files.

### 📊 Top 10 movies based on users ratings

![Top 10 movies based on users ratings](./Plots/Top%2010%20rated%20movies.png)

## 📂 Structure

```text
.
├── movies.csv                        # Movies Dataset
├── ratings.csv                       # User Ratings Dataset
├── IMDB Recommender System.ipynb     # Main ML Pipeline and Logic
├── item_based_knn_model.pkl          # Exported Production Model
├── movie_user_pivot_table.pkl        # Serialized Data Matrix
├── Plots                             # Project's plots
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

---

\- [Tony White](https://github.com/itstonywhite) ✍️
