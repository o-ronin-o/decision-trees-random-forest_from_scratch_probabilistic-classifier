import numpy as np

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_priors = {} 
        self.feature_log_likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # calculating class priors
        for c in self.classes:
            # getying all samples belonging to this class
            X_c = X[y == c]
            class_count = X_c.shape[0]
            
            # calculating prior with smoothing
            prior = (class_count + self.alpha) / (n_samples + self.alpha * num_classes)
            self.class_log_priors[c] = np.log(prior)

            # calculating features Likelihoods
            self.feature_log_likelihoods[c] = {}
            
            for feature_idx in range(n_features):
                # finding all unique values this feature can take
                unique_values = np.unique(X[:, feature_idx])
                num_feature_values = len(unique_values)
                self.feature_log_likelihoods[c][feature_idx] = {}

                for val in unique_values:
                    # counting how many times this specific value appears in this class
                    count_feature_val = np.sum(X_c[:, feature_idx] == val)
                    
                    likelihood = (count_feature_val + self.alpha) / (class_count + self.alpha * num_feature_values)
                    self.feature_log_likelihoods[c][feature_idx][val] = np.log(likelihood)

    def predict(self, X):

        predictions = []
        
        for i in range(len(X)):
            sample = X[i]
            class_scores = {}

            # calculating score for each class
            for c in self.classes:
                # starting with the Prior
                score = self.class_log_priors[c]
                
                # adding the likelihoods for each feature in the sample
                for feature_idx, val in enumerate(sample):
                    # but only add if this value was there during training
                    if val in self.feature_log_likelihoods[c][feature_idx]:
                        score += self.feature_log_likelihoods[c][feature_idx][val]
                    else:
                        pass 
                
                class_scores[c] = score
            
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
            
        return np.array(predictions)
