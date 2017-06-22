
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import numpy as np 
from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

data = pd.read_csv('C:/Users/vishnu.sk/Desktop/bot_training.csv')
data['User_Queries'] = data['User_Queries'].astype(str)
train_data_columns = data.columns.drop(['Label'])

X_train_counts = count_vect.fit_transform(data['User_Queries'])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X = X_train_tfidf.toarray()

x_train,x_test, y_train,y_test = train_test_split(X,
                                                  data['Label'],
                                                  random_state = 1)


result_cols = ["Classifier", "Accuracy"]
result_cols1 = ["Classifier","Precision", "Recall", 'F-measure']

result_frame = pd.DataFrame(columns=result_cols)
result_frame1 = pd.DataFrame(columns=result_cols1)

classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(),
        SGDClassifier(),
        LogisticRegression(multi_class='multinomial',solver ='newton-cg'),
        SVC(),
        GaussianNB()]


type1_error = []
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    acc = accuracy_score(y_test,predicted)
    precision = precision_score(y_test, predicted, average='weighted')
    rec = recall_score(y_test, predicted, average='weighted')
    f_measure = f1_score(y_test, predicted, average='weighted')
    type1_error.append([precision, rec, f_measure])

    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)
    
    acc_field1 = pd.DataFrame([[name, precision, rec, f_measure]], columns=result_cols1)
    result_frame1 = result_frame1.append(acc_field1)
    confusion_mc = confusion_matrix(y_test, predicted)
    df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])
    plt.figure(figsize=(5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title(name+'\nAccuracy:{0:.3f}'.format(acc))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plt.figure()
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

df1 = pd.melt(result_frame1, id_vars=['Classifier']).sort_values(['variable','value'])

plt.figure()
sns.barplot(x="Classifier", y="value", hue="variable", data=df1)
plt.xticks(rotation=90)
plt.show()


