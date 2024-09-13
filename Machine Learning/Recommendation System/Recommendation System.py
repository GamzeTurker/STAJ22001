import pandas as pd
movie = pd.read_csv(r"C:\Users\Gamze\Desktop\Python ile Yapay Zeka\ML\archive\movie.csv")
print(movie.columns)

movie=movie.loc[:,["movieId","title"]]
movie.head(10)

rating = pd.read_csv(r"C:\Users\Gamze\Desktop\Python ile Yapay Zeka\ML\archive\rating.csv")
rating.columns

rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)


data=pd.merge(movie,rating)
data.head()


data=data.iloc[:1000000,:]

# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)

movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()