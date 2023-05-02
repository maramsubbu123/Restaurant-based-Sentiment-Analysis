from google.colab import drive 
drive.mount('/content/gdrive')

"""**Importing Libraries**"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

df = pd.read_table('/content/gdrive/MyDrive/Verzeo/Restaurant_Reviews.tsv')#Copy the path of uploaded excel file from drive after mounting.
df

df.info()

df['Liked'].value_counts()

"""**Plotting the two classes of tweets**"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.bar(df['Liked'],500, width=0.3, color=['green','red'])
plt.show()

sns.pairplot(df)

df['Review'][412]

df['Liked'][412]

x = df['Review'].values
y = df['Liked'].values

"""**Splitting the dataset into train and test**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0, test_size=0.2)

x_train.shape

x_test.shape

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words = 'english')
x_train_vect = vect.fit_transform(x_train) 
x_test_vect = vect.transform(x_test)

"""**Model1 - (Support vector machine)(SVM)**"""

from sklearn.svm import SVC
model1 = SVC()

model1.fit(x_train_vect,y_train)

y_pred1 = model1.predict(x_test_vect)
y_pred1

y_test

"""**Model1 Accuracy, Precision, Recall, F1 - Score**"""

from sklearn.metrics import accuracy_score
print('Accuracy:',accuracy_score(y_pred1,y_test))
print('Precision: %.3f' % precision_score(y_test, y_pred1))
print('Recall: %.3f' % recall_score(y_test, y_pred1))
print('F1 Score: %.3f' % f1_score(y_test, y_pred1))

#to test the output 
test = vect.transform([df['Review'][412]]) 
model1.predict(test)

"""**Model2 - combines two estimators (countvect+svc)**"""

from sklearn.pipeline import make_pipeline 
model2 = make_pipeline(CountVectorizer(),SVC())

model2.fit(x_train,y_train)

y_pred2 = model2.predict(x_test)
y_pred2

from sklearn.metrics import accuracy_score
print('Accuracy:',accuracy_score(y_pred2,y_test))
print('Precision: %.3f' % precision_score(y_test, y_pred2))
print('Recall: %.3f' % recall_score(y_test, y_pred2))
print('F1 Score: %.3f' % f1_score(y_test, y_pred2))

model2.predict([df['Review'][412]])

"""**Model3 - Using Naive Bayes**"""

from sklearn.naive_bayes import MultinomialNB
model3 =MultinomialNB()

model3.fit(x_train_vect,y_train)

y_pred3 = model3.predict(x_test_vect)
y_pred3

y_test

"""**Model3 Accuracy, Precision, Recall, F1 - Score**"""

print('Accuracy:',accuracy_score(y_pred3,y_test))
print('Precision: %.3f' % precision_score(y_test, y_pred3))
print('Recall: %.3f' % recall_score(y_test, y_pred3))
print('F1 Score: %.3f' % f1_score(y_test, y_pred3))

# to evaluate a statement and see if its spam or not using the method3 
test = vect.transform([df['Review'][412]])
model3.predict(test)

"""**Model4 - Using pipeline(countvect,multinomialNB)**"""

from sklearn.pipeline import make_pipeline
model4 = make_pipeline(CountVectorizer(),MultinomialNB())
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
y_pred4

"""**Model4 Accuracy, Precision, Recall, F1 - Score**"""

print('Accuracy:',accuracy_score(y_pred4,y_test))
print('Precision: %.3f' % precision_score(y_test, y_pred4))
print('Recall: %.3f' % recall_score(y_test, y_pred4))
print('F1 Score: %.3f' % f1_score(y_test, y_pred4))

"""**Accuracy of differnt models after training:**

*   **ACCURACY FOR SVC - 73%**
*   **SVC PIPELINE - 79%**
*   **ACCURACY FOR MultinomialNB - 74.5%**
*   **MultinomialNB PIPELINE - 81%**

**Importing Joblib and dumping Model4**
"""

# joblib - persistance model  (used to save pipeline models)
import joblib
joblib.dump(model4,'Pos-Neg')

import joblib 
reload_model = joblib.load('Pos-Neg')

#predict using the reloaded joblib model 
reload_model.predict(["Crust is not good."])

"""**Installing the streamlit**"""

#install streamlit 
!pip install streamlit --quiet

# Commented out IPython magic to ensure Python compatibility.
# #STREAMLIT WEBAPP 
# %%writefile app.py 
# import streamlit as st  #webapp framework/library 
# import joblib 
# 
# reload_model = joblib.load('Pos-Neg')  #loads the joblib model 
# 
# st.title("Restaurant FeedBack")
# st.title("Food Castle")
# ip = st.text_input("Enter your feedback:") #asking the user input 
# 
# op = reload_model.predict([ip])  #predict the output 
# if st.button('PREDICT'): #if button is clicked 
#   st.title(op[0]) #prints the output in single dimension
#   if(op[0]==1):
#     st.write("Your review was Positive. Thanks for giving good feedback.")
#   else:
#     st.text_input("Your review was Negative. Kindly tell what we can improve")
#     st.write("Thanks for providing feedback.")

!streamlit run app.py & npx localtunnel --port 8501
