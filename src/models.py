from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Input

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate_model(y_test, y_pred), y_pred

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate_model(y_test, y_pred), y_pred

def train_cnn(X_train, y_train, X_test, y_test):
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)
    y_pred_prob = model.predict(X_test_reshaped)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    return model, evaluate_model(y_test, y_pred), y_pred, y_pred_prob

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }