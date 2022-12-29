import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus



df = pd.read_json('Film.JSON', orient='records')

# lets consider features including actor, genre

def genre_category(genre):
	if 'Action' in genre:
		return 3
	if 'History' in genre:
		return 3
	if 'Thriller' in genre:
		return 2
	if 'Fantasy' in genre:
		return 2
	return 1	

df['Genre Number Category'] = df["Genre"].map(genre_category)


def Has_actor(actors):
	if 'Matthew McConaughey' in actors:
		return 3
	if 'Leonardo DiCaprio' in actors:
		return 3
	if 'Kit Harington' in actors:
		return 3
	if 'Will Smith' in actors:
		return 2
	if 'Gerard Butler' in actors:
		return 2
	if 'Chris Evans' in actors:
		return 2
	return 1

df['Actors Number Category'] = df["Actors"].map(Has_actor)


features = ['Genre Number Category', 'Actors Number Category']

d = {'Yes': 1, 'No': 0}
df['Go'] = df['Go'].map(d)

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

#print(df[['Actors', 'Actors Number Category', 'Genre', 'Genre Number Category', 'Go']])

new_movies = [{
    "Title": "Titanic",
    "Genre": "Action, Romantic",
    "Director": "James Cameron",
    "Actors": "Leonardo DiCaprio, Kate Winslet",
    "Language": "English, Spanish",
    "Country": "USA, UK",
    "Type": "movie",
    "Response": "True",
  },
  {
    "Title": "The Purge",
    "Genre": "horror, fantasy",
    "Director": "James DeMonaco",
    "Actors": "Ethan Hawke",
    "Language": "English, Spanish",
    "Country": "USA, UK",
    "Type": "movie",
    "Response": "True",
  },
  {
    "Title": "The Gentlemen",
    "Genre": "Action, Comedy",
    "Director": "Guy Ritchie",
    "Actors": "Matthew McConaughey",
    "Language": "English, Spanish",
    "Country": "USA, UK",
    "Type": "movie",
    "Response": "True",
  }
  ]

df2 = pd.DataFrame(new_movies)
df2['Genre Number Category'] = df2["Genre"].map(genre_category)
df2['Actors Number Category'] = df2["Actors"].map(Has_actor)

print(dtree.predict(df2[features]))

y_test = [1,0,1]

print("Accuracy:",metrics.accuracy_score(y_test, dtree.predict(df2[features])))

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())

# Random tree can capture non-linear relationship but prone to noise(overfitting)
# Random forest is used to overcome overfitting data. Split the dataset, using many decision tree to produce outputs. 
# Which ouput has a majority from all decision trees will be picked.