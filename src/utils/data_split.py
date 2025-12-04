from sklearn.model_selection import train_test_split




def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=seed)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_ratio, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test