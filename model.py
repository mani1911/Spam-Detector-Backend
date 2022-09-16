import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('spam.csv', encoding="latin1")
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.2)

v= CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

data = {"model":clf}
with open('model_pickle.pkl', 'wb') as file:
    pickle.dump(data, file)


