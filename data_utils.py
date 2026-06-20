from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Constants
RANDOM_STATE = 42
TEST_FRACTION = 0.30
MODEL_PATH = "savedmodel.pth"
IMAGE_SIZE = 64

# function to load dataset
def load_face_dataset():
    """Returns the faces feature matrix (X) and target labels (Y)"""
    bundle = fetch_olivetti_faces(shuffle = True, random_state = RANDOM_STATE)
    return bundle.data, bundle.target

# function to split 70/30
def make_train_test_split():
    """Split into a startified 70/30 train/test partiotion"""
    features, labels = load_face_dataset()
    return train_test_split(
        features,
        labels,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels
    )
