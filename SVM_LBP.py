from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load grayscale image
image = cv2.imread("sandalwood_leaf.jpg", cv2.IMREAD_GRAYSCALE)

# Compute LBP
lbp = local_binary_pattern(image, P=8, R=1, method="uniform")

# Flatten LBP features
X = lbp.flatten().reshape(1, -1)  # Reshape for classifier

# Dummy labels (Healthy = 0, Infected = 1)
y = np.array([1])

# Train a simple SVM classifier (Use real dataset for better results)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")