# Kaggle_HomeDepotSearchRelevance
Prediction of search similarity score measure using Linear Regression Model on Home Depot Products, data provided by Kaggle

# Features:
	The features are the attributes of a particular product. 
	The attributes of high frequencies are ‘brand’,’material’,’dimension’,’color’,’bullets’,’product height’,’product weight’, ’width’, title and description
# Stem:
The description of a product is stemmed and the stop words are removed. The same is done for all the features of a product.

# Implementation technique:
The features, description, title of a product are added to a dictionary. The attributes to ‘attrmapping_dict’.Dictionary ‘attrcategories_index’ 
Indexes the attributes in ‘attrmapping_dict’.A dictionary of all the products , ‘filedictionary’ contains key as the product id and the value being the attributes ‘attrmapping_dict’. 

# Tf-idf and cosine similarity:
The cosine similarity between the search term and feature of every product id is stored in the array ‘productcategoryvector’. This is returned on the function call of ‘queryprod’ (all function methods will be explained in detail later in the doc). 

# Model:
After calculating the cosine similarity between the search term and the features for a corresponding product id , model is applied. I used linear regression.

First formed the matrix using numpy library. 

Xij = cosine similarity between the search term and each f product feature I, j. 
i = each row
j = each feature column
Search(i) = search term for a product
Yi = relevance score of a product ith row

# Linear regression:
	Linear regression attempts to create model between 
Two variables by fitting a linear equation to observed data. In our project it is multivariate where the Xij features are the variable and is equated to the relevance in the train data set ‘Yi’ forming a linear equation.

	Y = mX + b
 
In our project it is

	Yi = Xij + theta , we consider theta = differs
 Or
	Y1 = X11 + X12 + X13 + …..X1n + 1

The x is called as the predictor variable and y as the criterion variable. Line regression finds the best fitting straight line through the points so as to lower the error values. I used gradient descent to find the best minimal error value. It starts with a initial parameter and iteratively moves towards finding the minimal error value.

Error = i/n sum(sqr (Yi – (Xij + theta)))
