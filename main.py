import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#load dataset
data=pd.read_csv("spam.csv")

#Rename columns
data.columns = ['label', 'message']

#convert labels into labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

#Show processed dataset
print("Processed Data:")
print(data.head())

#convert text into numbers
cv =CountVectorizer()
X = cv.fit_transform(data['message'])

print("\nText Converted Into Numbers:")
print("\nText successfully converted into numerical format.")

#labels
y = data['label']

#split dataset into training and testing
X_train, X_test, y_train, y_test= train_test_split(
    X,y, test_size=0.2, random_state=42
)

#Create model
model = MultinomialNB()
     
#Train model
model.fit(X_train, y_train)

print("\nModel trained successfully")

#check accuracy
accuracy = model.score(X_test,y_test)
print("\nAccuracy:", accuracy * 100, "%")

#predict new email
msg = [input("Enter an email message: ")]

msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("This email is spam")
else:
    print("This email is Not spam")