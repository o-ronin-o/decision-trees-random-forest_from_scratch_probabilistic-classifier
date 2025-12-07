import numpy as np

class GaussianGenerativeClassifier:
    def __init__(self, regularization=1e-1):
        
        self.lambd = regularization
        self.priors = None  
        self.means = None   
        self.cov_inv = None
        self.classes = None 

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features)) # 10 x 64 matrix 
        
        for i, c in enumerate(self.classes):
            # Filter samples belonging to the class 
            X_c = X[y == c] # x now has the images of current class only
            
            self.priors[i] = X_c.shape[0] / n_samples
            
            self.means[i] = np.mean(X_c, axis=0)

        
        X_centered = X - self.means[y] 
        covariance = (X_centered.T @ X_centered) / n_samples
        # Sigma_lambda = Sigma + lambda * I
        cov_regularized = covariance + (self.lambd * np.eye(n_features))
        
        self.cov_inv = np.linalg.inv(cov_regularized)

    def predict(self, X):
        scores = []
        for x in X:
            class_scores = []

            for i in range(len(self.classes)):
                log_prior = np.log(self.priors[i])
                
                # distance = -0.5 * diff ^T * coveriance-inv * diff
                diff = x - self.means[i]
                distance = diff.T @ self.cov_inv @ diff
                log_likelihood_term = -0.5 * distance
                
                total_score = log_prior + log_likelihood_term
                class_scores.append(total_score)
            
            scores.append(class_scores)
        
        predictions = np.argmax(scores, axis=1)
        return self.classes[predictions]