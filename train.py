import joblib
from sklearn.tree import DecisionTreeClassifier

from data_utils import make_train_test_split, MODEL_PATH, RANDOM_STATE


# training function
def build_and_train():
    X_train, X_test, Y_train, Y_test = make_train_test_split()
    print(f"Loaded -> train: {X_train.shape[0]} samples, "
          f"test: {X_test.shape[0]} samples, features: {X_train.shape[1]}")

    # Create and train model
    classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, Y_train)

    train_accuracy = classifier.score(X_train, Y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # save the model
    joblib.dump(classifier, MODEL_PATH)
    print(f"Saved trained model to '{MODEL_PATH}'")
    return classifier


if __name__ == "__main__":
    build_and_train()
