import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.naive_bayes import NaiveBayesClassifier

class TestNaiveBayes(unittest.TestCase):
    
    def setUp(self):
        self.X = np.array([
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0]
        ])
        self.y = np.array([0, 0, 1, 1])
        
        self.model = NaiveBayesClassifier(alpha=1.0)

    def test_fit_runs(self):
        try:
            self.model.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"fit() raised Exception unexpectedly: {e}")

    def test_predictions_shape(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_simple_logic(self):
        X_simple = np.array([[0], [0], [1], [1]])
        y_simple = np.array([0, 0, 1, 1])
        
        self.model.fit(X_simple, y_simple)
        
        test_data = np.array([[0], [1]])
        preds = self.model.predict(test_data)
        
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)

if __name__ == '__main__':
    unittest.main()