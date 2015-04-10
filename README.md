# STAT642-Statistical_Computing
A place to share, organize, and discuss code for 2015 Spring quarter offering of STAT 642 - Statistical Computing

I solved the problem with not enough iterations for regression.  With Lasso and ElasticNet (SKLearn) you can add in an argument in the same place where the alpha setting is, like this:

.Lasso(alpha=0.5, max_iter=10000)

Note that the .Lasso comes after you call the function, such as linear_model (or whatever you named it when you imported from SKLearn).
