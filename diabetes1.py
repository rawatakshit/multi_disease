import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


# Load data
data = pd.read_csv("Projects/diabetes2.csv")

# Encode 'gender' column (Male -> 1, Female -> 0, Other -> 2)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

# Separate features and target
X = data.drop(columns=["diabetes", "smoking_history"], axis=1)
Y = data["diabetes"]

# Standardize numerical features
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, Y_train)

# Make predictions
train_predictions = rf_classifier.predict(X_train)
test_predictions = rf_classifier.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score( train_predictions,Y_train)
test_accuracy = accuracy_score( test_predictions,Y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Save the model
pickle.dump(rf_classifier, open('./diabetes_model_with_gender.sav', 'wb'))
# Save the fitted StandardScaler
pickle.dump(scaler, open('./scaler.sav', 'wb'))
