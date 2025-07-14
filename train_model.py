import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('dataset.csv')

# Clean column names to avoid issues with extra spaces
data.columns = data.columns.str.strip()

# Convert 'gender' to numerical values: 'male' = 1, 'female' = 0
data['gender'] = data['gender'].map({'male': 1, 'female': 0})

# Check for missing 'LungCancer' column
if 'LungCancer' not in data.columns:
    print("Error: 'LungCancer' column is missing in the dataset.")
    exit()

# Features (everything except target 'LungCancer')
X = data.drop('LungCancer', axis=1)
# Target
y = data['LungCancer']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save trained model
with open('lung_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save accuracy info (optional)
with open('model_accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
