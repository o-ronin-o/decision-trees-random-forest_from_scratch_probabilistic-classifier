# Custom metrics implementation (precision, recall, F1, etc.)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error





def calculate_metrics(y_true, y_pred, label = ''):
    metrics = {
                f'{label}accuracy': accuracy_score(y_true, y_pred),
                f'{label}precision': precision_score(y_true, y_pred),
                f'{label}recall': recall_score(y_true, y_pred),
                f'{label}f1_score': f1_score(y_true, y_pred),
                f'{label}confusion_matrix': confusion_matrix(y_true, y_pred)
            }
    return metrics


def analyze_tree_complexity(model):
    """Analyze tree structure and complexity"""
    n_nodes = model.no_nodes
    n_leaves = model.no_leaves
    depth = model.max_reached_depth

    
    complexity_metrics = {
        'Number of Nodes': n_nodes,
        'Number of Leaves': n_leaves,
        'Tree Depth': depth,
        'Branching Factor': (n_nodes - n_leaves) / (n_nodes - 1) if n_nodes > 1 else 0
    }
    
    return complexity_metrics