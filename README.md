# pregoogle
A simple web app that uses ML to predict if a recipe is safe or not safe for pregnant woman. 

## Problem 
1. According to the National Center for Health care Statistics (NCHS), in 2016, there were approximately 1.1 million fetal losses in the United States. Approximately 10% of those are directly attributed to infectious diseases.
2. Popular online recipe review sites have begun to highlight allergy/gluten restrictions on their recipes. However, they do not yet have a feature that speaks to the unique dietary restrictions of pregnant women. 
3. It’s difficult to identify whether a recipe is pregnancy-safe or not from simply scanning the ingredient list because: 
    * There are a large number of unsafe foods.
    * Certain foods are only unsafe based on style of preparation.

## Solution
Websites could start placing icons beside recipes not recommended for pregnant women. 

In this prototype, I created a search site that will scrape AllRecipes.com based on the recipe ID to determine if the food is safe. The presentation can be seen here:

https://docs.google.com/presentation/d/1r1hnA1MNYzTb8FjMDCmpkGqsSh0F2YmVTOS1Nki4gXY/edit#slide=id.p3

## Implmentation overview
1. Use python to scrape reviews from allrecipes.com
2. Manually label a few of these reviews
3. Clean data by eliminating pronouns, punctuation, etc.
4. Transform string into token count
5. Transform toekn count into TF-IDF score vectors
6. Train on TF-IDF vectors with Naive Bayes 
7. Other techniques applied for optimization
  * Cross validation
  * Grid search
  
## Results

![](https://firebasestorage.googleapis.com/v0/b/test-840a6.appspot.com/o/confusion.png?alt=media&token=038657dd-97de-4c52-8d2d-41f750e0d89b)
![](https://firebasestorage.googleapis.com/v0/b/test-840a6.appspot.com/o/roc.png?alt=media&token=ba4b64ef-97de-4860-84fc-e46bbea7c0bb)
