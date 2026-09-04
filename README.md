# NN_EXP-6
<H3>ENTER YOUR NAME :POOJA PRIYA B</H3>
<H3>ENTER YOUR REGISTER NO. 212224230196</H3>
<H1>EX. NO.6</H1>
<H1 >Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>
<pre>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv("/content/drive/MyDrive/heart.csv")

features = dataset.iloc[:, :-1].values
target = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)


model = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    max_iter=1000,
    random_state=42
)

model.fit(x_train, y_train)

loss_values = model.loss_curve_

predicted_output = model.predict(x_test)

acc = accuracy_score(y_test, predicted_output)
print("Accuracy:", acc)

plt.figure(figsize=(8, 5))
plt.plot(loss_values)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()
cm = confusion_matrix(y_test, predicted_output)
cr = classification_report(y_test, predicted_output)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)
</pre>

<H3>Output:</H3>

<img width="967" height="750" alt="image" src="https://github.com/user-attachments/assets/bcad5a9a-57e9-4acb-832f-ff9c5d13efdd" />


<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
