# ML Assignment – Probabilistic Models, Decision Trees & Random Forests

This repository contains complete from-scratch implementations of:

- Gaussian Generative Classifier (LDA-style)
- Gaussian Naive Bayes
- Decision Tree Classifier for continuous features
- Random Forest using custom decision trees

The project follows a full ML pipeline: dataset preparation, stratified splitting,
hyperparameter tuning, evaluation, metrics, visualization, and model comparison.



---



## 🔍 Implemented Models

### **1. Gaussian Generative Classifier**
- Estimates class priors, means, shared covariance
- Uses regularized Σ + λI
- Tuned over λ ∈ {1e−4, 1e−3, 1e−2, 1e−1}

### **2. Naive Bayes (Categorical)**
- Laplace smoothing α ∈ [0.1, 5]
- Comparisons with sklearn’s MultinomialNB
- Full probability-table implementation

### **3. Decision Tree (from scratch)**
- Continuous feature splits
- Entropy & Information Gain
- Hyperparameters:
  - max_depth ∈ {2,4,6,8,10}
  - min_samples_split ∈ {2,5,10}
- Feature importance via accumulated information gain

### **4. Random Forest (Bonus)**
- Bootstrap sampling
- Random feature subsets
- Majority voting
- Hyperparameters:
  - T ∈ {5,10,30,50}
  - max_features ∈ {sqrt(d), d/2}

---

## 📊 Evaluation

Each model includes:

- Accuracy, Precision, Recall, F1
- Confusion matrix
- Cross-model comparison
- Bias–variance analysis (Tree vs Forest)

---

## 🧠 Why This Project is Interesting

This project demonstrates:

- Complete ML model implementations **without sklearn**  
- Understanding of entropy, information gain, and tree construction  
- Ensemble learning and variance reduction  
- Real-world datasets (Digits, Adult, Breast Cancer)

This builds skills in:
- Mathematical modeling  
- Core ML algorithms    
- Experimental analysis  

