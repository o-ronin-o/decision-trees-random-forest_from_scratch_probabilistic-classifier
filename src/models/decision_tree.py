import numpy as np 
from collections import Counter
import pandas as pd

class Node:
    def __init__(self, feature = None, threshold = None,left = None, right = None,*,val = None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.val = val

    def is_leaf_node(self):
        return self.val is not None
    
    
    def value(self):
        return self.val
    

class DecisionTree:
    def __init__(self, min_samples = 2, max_depth = 100, n_features=None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.features_importance = []
        self.no_nodes = 0
        self.no_leaves = 0
        self.max_reached_depth = 0
        

    # def _add_feature_importance(self,label = '', gain = None):
    #     self.features_importance.append({"Feature Name" : label , "Gain": gain})

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)



    def _grow_tree(self,X,y,depth=0):
        # getting required info from data 
        n_samples,n_features =  X.shape
        n_labels = len(np.unique(y))
        self.max_reached_depth = max(self.max_reached_depth, depth)

        #stoping criteria 
        """
        - if we exceed max depth
        - if we have a single label (pure)
        - if we have n_samples less than minimum number of samples decided at initialization 

        action: 
            - we specify the label and return a node 
        """
        if(depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples):
            value = self._most_common_label(y)
            self.no_leaves += 1
            self.no_nodes +=1 
            return Node(val=value)

        # we shall provide optional randomness in case we extend to random forest

        feat_idxs= np.random.choice(n_features,self.n_features, replace = False)

        # getting best split(best feature and best split)
        
        best_threshold,best_feature = self._best_split(X,y,feat_idxs)

        # now we split     
        left_idxs,right_idxs = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs], depth+1)
        

        self.no_nodes +=1
        return Node(best_feature,best_threshold,left,right)
    
    def calculate_feature_importance(self, feature_names=None):
        """
        Calculate feature importance by aggregating information gain for each feature.
        """
        # Extract feature names and Gains
        features = [n["Feature Name"] for n in self.features_importance]
        gains = [n["Gain"] for n in self.features_importance]

        # Aggregate gains per feature
        df = pd.DataFrame({
            "feature": features,
            "gain": gains
        })

        importance_df = (
            df.groupby("feature")["gain"]
            .sum()                      # total gain
            .sort_values(ascending=False)
            .reset_index()
        )

        importance_df.columns = ["feature", "importance"]
        return importance_df


    
    # we predict by traversing the tree we created 
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    

    # a recursive function to traverse the tree 
    def _traverse_tree(self,x, node):
        if node.is_leaf_node():
            return node.value()
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)


    def _most_common_label(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label 

    def _best_split(self,X,y,feat_idxs):
        best_gain = -1 
        split_threshold, split_idx = None, None
        local_importances = []

        # i loop on all features 
        for idx in feat_idxs:
            x_column = X[:,idx]
            thresholds = np.unique(x_column)
            best_feature_gain = -1
            # for each feature unique threshold we calculate gain 
            for thr in thresholds:
                gain = self._information_gain(y , x_column, thr)

                if gain > best_feature_gain:
                    best_feature_gain = gain

                # better gain? switch!
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx 
                    split_threshold = thr
            local_importances.append({"Feature Name": f"Feature {idx}", "Gain": float(best_feature_gain)})
        self.features_importance.extend(local_importances)

        return split_threshold, split_idx


    """ 
    to get information gain we need: 
        - calculated entropy of the parent
        - create children
        - calculate weighted entropy of children 
        - calculate IG
    """
    def _information_gain(self, y, x_column, thr):
        parent_entropy = self._entropy(y)

        #create children 
        left_idx, right_idx = self._split(x_column , thr) 

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # now we calculate the weighted entropy 
        # we need the # of samples, # of right and left samples
        # we need to calculate the entropy of children 

        n = len(y)
        n_l,n_r = len(left_idx), len(right_idx)
        e_l,e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx]) 

        weighted_entropy_child = (n_l/n) * e_l + (n_r/n) * e_r

        # calculating IG
        information_gain =  parent_entropy - weighted_entropy_child
        
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log(p) for p in ps if p > 0])
    

    def _split(self, x_column, split_threshold):
        left_idx = np.argwhere(x_column <= split_threshold).flatten()
        right_idx = np.argwhere(x_column > split_threshold).flatten()    
        return left_idx,right_idx