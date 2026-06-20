import os
import sys

import joblib
from sklearn.metrics import accuracy_score

from data_utils import make_train_test_split, MODEL_PATH

# evaluate function
def evaluate(): 
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[error] '{MODEL_PATH}' not found - run 'python train.py' first.")

    # get the test set    
    _, X_test, _, y_test = make_train_test_split()

    # load the model
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    return accuracy

if __name__ == "__main__":
    evaluate()
