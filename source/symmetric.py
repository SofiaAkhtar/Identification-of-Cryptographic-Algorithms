import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import chisquare
import binascii
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
def load_dataset(filename):
    data = pd.read_csv(filename)
    return data['Ciphertext'], data['Algorithm']

# Feature extraction functions
def number_of_unique_characters(byte_data):
    return len(set(byte_data))

def chi_squared_statistic(byte_data):
    # Frequencies of English letters A-Z (both upper and lowercase)
    english_freq = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 0.002,
                    0.008, 0.040, 0.024, 0.067, 0.075, 0.019, 0.001, 0.060, 0.063, 0.091,
                    0.028, 0.010, 0.023, 0.001, 0.020, 0.001]  # English letter frequency

    byte_freq = np.zeros(26)  # Only letters A-Z/a-z
    total_letters = 0

    # Ensure byte_data is integer array
    byte_data = byte_data.astype(np.uint8)

    # Filter byte_data to only keep letters (A-Z or a-z)
    for byte in byte_data:
        if 65 <= byte <= 90:  # A-Z
            byte_freq[byte - 65] += 1
            total_letters += 1
        elif 97 <= byte <= 122:  # a-z
            byte_freq[byte - 97] += 1
            total_letters += 1

    if total_letters == 0:
        return 0  # No alphabetic characters, return 0

    # Normalize byte frequencies
    byte_freq /= total_letters
    byte_freq *= sum(english_freq)

    # Perform chi-square test
    return chisquare(byte_freq, english_freq)[0]

def index_of_coincidence(byte_data):
    byte_data = byte_data.astype(np.uint8)  # Ensure byte_data is integer
    freqs = np.bincount(byte_data)
    N = len(byte_data)
    return sum(f * (f - 1) for f in freqs) / (N * (N - 1)) if N > 1 else 0

def max_ic_periods(byte_data, max_period=15):
    ic_values = []
    for p in range(1, max_period + 1):
        shifted = [byte_data[i::p] for i in range(p)]
        avg_ic = np.mean([index_of_coincidence(s) for s in shifted])
        ic_values.append(avg_ic)
    return max(ic_values)

def max_kappa_periods(byte_data, max_period=15):
    kappa_values = []
    for p in range(1, max_period + 1):
        shifted = np.roll(byte_data, p)
        kappa = np.mean(byte_data == shifted)
        kappa_values.append(kappa)
    return max(kappa_values)

def digraphic_index_of_coincidence(byte_data):
    pairs = [byte_data[i:i+2] for i in range(0, len(byte_data) - 1, 2)]
    pair_freqs = np.bincount([pair[0] * 256 + pair[1] for pair in pairs], minlength=256 * 256)
    N = len(pairs)
    return sum(f * (f - 1) for f in pair_freqs) / (N * (N - 1)) if N > 1 else 0

def long_repeat(byte_data):
    trigrams = [byte_data[i:i+3] for i in range(len(byte_data) - 2)]
    trigram_freqs = np.bincount([int.from_bytes(t, 'big') for t in trigrams], minlength=256**3)
    return np.sqrt(np.mean(trigram_freqs > 1))

def entropy(byte_data):
    freqs = np.bincount(byte_data) / len(byte_data)
    return -np.sum(freqs[freqs > 0] * np.log2(freqs[freqs > 0]))

def pad_or_chunk_ciphertext(byte_data, max_len=512):
    if len(byte_data) > max_len:
        # Chunk the data if it's larger than max_len
        return byte_data[:max_len]  # Truncate or chunk to max_len
    else:
        # Pad the data to make it max_len
        padded_data = pad_sequences([byte_data], maxlen=max_len, padding='post', dtype='uint8')[0]
        return padded_data

# Extract features from ciphertext
def extract_features_from_hex(hex_data):
    byte_data = np.frombuffer(binascii.unhexlify(hex_data), dtype=np.uint8)
    byte_data = pad_or_chunk_ciphertext(byte_data)  # Apply padding or chunking
    
    features = [
        number_of_unique_characters(byte_data),              # NUC
        chi_squared_statistic(byte_data),                    # CSS
        index_of_coincidence(byte_data),                     # IC
        max_ic_periods(byte_data),                           # MIC
        max_kappa_periods(byte_data),                        # MKA
        digraphic_index_of_coincidence(byte_data),           # DIC
        long_repeat(byte_data),                              # LR
        entropy(byte_data)                                   # ENT
    ]
    return np.array(features)

# Load and prepare training data
ciphertexts, labels = load_dataset("cipher_dataset.csv")

# Extract features for each ciphertext in the dataset
X = np.array([extract_features_from_hex(ct) for ct in ciphertexts])
y = np.array(labels)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "SVM": CalibratedClassifierCV(SVC(kernel='linear', probability=True, random_state=42))
}

# Train the classifiers and collect their predictions
train_preds = []
test_preds = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_preds.append(clf.predict_proba(X_train))
    test_preds.append(clf.predict_proba(X_test))
    print(f"{name} classifier trained.")

# Combine the predictions into a new feature set for the meta-classifier
X_train_meta = np.hstack(train_preds)
X_test_meta = np.hstack(test_preds)

# Train the final identification model (SVM) on the output probabilities of the classifiers
meta_classifier = SVC(kernel='linear', probability=True, random_state=42)
meta_classifier.fit(X_train_meta, y_train)

# Predict and evaluate the meta-classifier on the test set
y_pred = meta_classifier.predict(X_test_meta)
print("Meta-classifier (SVM) Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Meta-classifier Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Feature importance from RandomForest (for example)
feature_names = ['NUC', 'CSS', 'IC', 'MIC', 'MKA', 'DIC', 'LR', 'ENT']
importances = classifiers["RandomForest"].feature_importances_
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")

# Function to predict the cryptographic algorithm from a hexadecimal ciphertext
def predict_algorithm_and_compare_with_dataset(hex_ciphertext):
    # Verify and clean input
    cleaned_input = hex_ciphertext.replace(" ", "").strip()
    if not all(c in "0123456789abcdefABCDEF" for c in cleaned_input) or len(cleaned_input) % 2 != 0:
        raise ValueError("Invalid hexadecimal input. Ensure the input is valid hexadecimal data and has an even length.")

    # Extract features from the input ciphertext
    input_features = extract_features_from_hex(cleaned_input)
    input_features_scaled = scaler.transform([input_features])  # Scale features

    # Predict using classifiers and meta-classifier
    classifier_probs = [clf.predict_proba(input_features_scaled) for clf in classifiers.values()]
    combined_probs = np.hstack(classifier_probs)
    
    predicted_class = meta_classifier.predict(combined_probs)[0]
    probabilities = meta_classifier.predict_proba(combined_probs)[0]

    # Compare the input ciphertext's features with the dataset's features
    distance_to_classes = np.linalg.norm(X_scaled - input_features_scaled, axis=1)
    closest_idx = np.argmin(distance_to_classes)
    closest_algorithm = y[closest_idx]

    return predicted_class, probabilities, closest_algorithm, distance_to_classes[closest_idx]

# Ask the user for input
user_input = input("Enter the hexadecimal ciphertext: ")

# Validate input (basic check for hex characters)
try:
    cleaned_input = user_input.replace(" ", "").strip()

    if all(c in "0123456789abcdefABCDEF" for c in cleaned_input) and len(cleaned_input) % 2 == 0:
        predicted_algorithm, probabilities, closest_algorithm, distance = predict_algorithm_and_compare_with_dataset(cleaned_input)
        print(f"Predicted Cryptographic Algorithm: {predicted_algorithm}")
        print("Probabilities for each class:")
        for algo, prob in zip(meta_classifier.classes_, probabilities):
            print(f"{algo}: {prob:.4f}")
        
        print(f"Closest Algorithm in the Dataset: {closest_algorithm}")
        print(f"Distance to Closest Algorithm's Features: {distance:.4f}")
    else:
        print("Invalid hexadecimal input. Please ensure the input is valid hexadecimal data and has an even length.")
except Exception as e:
    print(f"An error occurred: {e}")
