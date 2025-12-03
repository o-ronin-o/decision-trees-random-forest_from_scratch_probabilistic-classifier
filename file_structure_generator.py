import os

# ------------------------------
# Folder + file definitions
# ------------------------------

PROJECT_NAME = "ml-assignment-3"

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "src/utils",
    "src/models",
    "src/training",
    "src/evaluation",
    "notebooks",
    "tests"
]

FILES = {
    "README.md": "# Machine Learning Assignment 3\n\nGenerated project structure.\n",
    "requirements.txt": "numpy\npandas\nscikit-learn\nmatplotlib\n",
    "run_all.py": "# Entry point for running all experiments\n\nif __name__ == '__main__':\n    print('Run each training script separately.')\n",
}

# Template minimal python modules
PYTHON_PLACEHOLDERS = {
    "src/utils/data_split.py": "# Utility functions for dataset splitting\n",
    "src/utils/metrics.py": "# Custom metrics implementation (precision, recall, F1, etc.)\n",
    "src/utils/plots.py": "# Plotting utilities\n",

    "src/models/gaussian_generative.py": "# Gaussian Generative Classifier implementation\n",
    "src/models/naive_bayes.py": "# Naive Bayes implementation\n",
    "src/models/decision_tree.py": "# Decision Tree implementation\n",
    "src/models/random_forest.py": "# Random Forest implementation\n",

    "src/training/train_gaussian.py": "# Training script for Gaussian Generative Model\n",
    "src/training/train_naive_bayes.py": "# Training script for Naive Bayes\n",
    "src/training/train_decision_tree.py": "# Training script for Decision Tree\n",
    "src/training/train_random_forest.py": "# Training script for Random Forest\n",

    "src/evaluation/evaluate_gaussian.py": "# Evaluation script for Gaussian Generative Model\n",
    "src/evaluation/evaluate_naive_bayes.py": "# Evaluation script for Naive Bayes\n",
    "src/evaluation/evaluate_decision_tree.py": "# Evaluation script for Decision Tree\n",
    "src/evaluation/evaluate_random_forest.py": "# Evaluation script for Random Forest\n",

    "tests/test_tree.py": "# Unit tests for Decision Tree\n",
    "tests/test_nb.py": "# Unit tests for Naive Bayes\n",
}

def create_directories():
    for d in DIRECTORIES:
        path = os.path.join(PROJECT_NAME, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

        # Add __init__.py to make it a Python package
        if "src" in d:
            init_path = os.path.join(path, "__init__.py")
            with open(init_path, "w") as f:
                f.write("# Package initializer\n")
            print(f"  Added: {init_path}")

def create_files():
    for filename, content in FILES.items():
        path = os.path.join(PROJECT_NAME, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created file: {path}")

def create_python_modules():
    for filename, content in PYTHON_PLACEHOLDERS.items():
        path = os.path.join(PROJECT_NAME, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created python module: {path}")

# ------------------------------
# Main execution
# ------------------------------

if __name__ == "__main__":
    print(f"\n📁 Generating project structure: {PROJECT_NAME}\n")

    os.makedirs(PROJECT_NAME, exist_ok=True)

    create_directories()
    create_files()
    create_python_modules()

    print("\n✅ Project structure created successfully!\n")
