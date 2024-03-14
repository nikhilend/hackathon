import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Assuming you have a dataset where the last column indicates whether the customer accepted the personal loan or not
# Adjust the path to your dataset
#data = pd.read_excel('personalloan.xlsx', sheet_name=0)

# Assuming your features are all columns except the last one
#X = data.iloc[:, :-1]
#y = data.iloc[:, -1]

# Splitting the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (you can use any other classifier as well)
#model = RandomForestClassifier()

# Training the model
#model.fit(X_train, y_train)

# Predicting on the testing set
#y_pred = model.predict(X_test)

# Calculating accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Save the model as a .pkl file
#joblib.dump(model, 'personalloan.pkl')

# Load the saved model
model = joblib.load('personalloan.pkl')

# Assuming you have new data in a DataFrame format
# Replace 'new_data.csv' with the path to your new data file
new_data = pd.read_csv('input.csv')

# Make predictions on the new data
predictions = model.predict(new_data)
# Display the predictions
print(predictions)