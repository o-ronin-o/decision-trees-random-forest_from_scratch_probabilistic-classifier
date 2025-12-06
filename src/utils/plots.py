# Plotting utilities
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes=[str(i) for i in range(10)]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()
    return cm


def decision_tree_vs_performance(min_samples_splits , results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, min_split in enumerate(min_samples_splits):
        ax = axes[idx]
        df = results[f'min_samples_split={min_split}']
        
        ax.plot(df['max_depth'], df['train_accuracy'], 'b-o', label='Training Accuracy')
        ax.plot(df['max_depth'], df['validation_accuracy'], 'r-o', label='Validation Accuracy')
        ax.fill_between(df['max_depth'], df['train_accuracy'], df['validation_accuracy'], 
                        alpha=0.2, color='gray')
        
        ax.set_xlabel('Max Depth')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'min_samples_split = {min_split}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()