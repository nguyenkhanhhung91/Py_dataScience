import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

import streamlit as st

st.title ("This App can classify whether a movie should be in your watch list. ")
st.header("It uses the decision tree model.")


df = pd.read_json('Data_ForStreamLit/Film.JSON', orient='records')

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

st.write("Please enter the information about a new movie and we will let you know if you want to watch it based on your judgement in the past")

title = st.text_input('Movie title')

Genre = st.multiselect("Enter your new movie genre:",["Action", "History", "Thriller", "Fantasy"])

director = st.selectbox("Enter your new movie director:",["James Cameron", "Guy Ritchie", "Chris Nolan", "Other"])

Actors = st.multiselect("Enter actors from new movie:",["Matthew McConaughey", "Leonardo DiCaprio", "Kit Harington", "Will Smith" ,"Gerard Butler", "Chris Evans" ])


if st.button('Predict'):
    # test prediction with new movies 
    new_movies = [{
        "Title": title,
        "Genre": Genre,
        "Director": director,
        "Actors": Actors,
        "Language": "English, Spanish",
        "Country": "USA, UK",
        "Type": "movie",
        "Response": "True",
      }
    ]

    df2 = pd.DataFrame(new_movies)
    df2['Genre Number Category'] = df2["Genre"].map(genre_category)
    df2['Actors Number Category'] = df2["Actors"].map(Has_actor)


    st.write("Here is the prediction")

    if (dtree.predict(df2[features])) == 0:
        st.write("You won't like it. Please see the tree below for explanation")
    else: 
        st.write("You will sure like it. Please see the tree below for explanation")

    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = features,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('tree.png')
    Image(graph.create_png())
    st.image("tree.png")


st.write('''
The math behind each decision is based on Gini Impurity or entropy/information gain. 
We can use "information gain" to determine how good the splitting of nodes in a decision tree.
The more the entropy removed, the greater the information gain. The higher the information gain, the better the split.
More at https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
Random tree can capture non-linear relationship but prone to noise(overfitting)
Random forest is used to overcome overfitting data. Split the dataset, using many decision tree to produce outputs. 
Which ouput has a majority from all decision trees will be picked.
More on Gini and decision tree: https://www.youtube.com/watch?v=_L39rN6gz7Y
More on Entropy and the math behind decision tree https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
	''')
