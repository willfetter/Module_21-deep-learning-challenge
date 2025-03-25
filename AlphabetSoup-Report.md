# Module 20 Report Template

## Overview of the Analysis

The purpose of this analysis was to develop a tool for the Alphabet Soup nonprofit foundation that can help it select the applicants for funding with the best chance of success in their ventures. It uses the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

The starter data was located within a "charity_data.csv" file and contained the loan status which was either "1 for a succesful organization or "0" for an unscuccesful organization. The variables we were trying to predict with the models were how many organizations are predicted to be successful if funded by Alphabet Soup. 

The model is built using the dependent variable "y" as 'IS_SUCCESSFUL', while the independent variable "x" represents all other data including 'APPLICATION TYPE', 'AFFILIATION", 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', and'ASK_AMT'.

## Analysis Results


### Data Preprocessing
#### What variable(s) are the target(s) for your model?
       
 - The target variable is 'IS_SUCCESSFUL' as this is a binary variable that indicates whether an organization was successful or not. '0' represents not successful, '1' represents successful.

#### What variable(s) are the features for your model?

 - The feature variables are all of the other information found within the dataset, except the 'EIN' and 'NAME'. Includes 'APPLICATION TYPE', 'AFFILIATION", 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', 'ASK_AMT')

#### What variable(s) should be removed from the input data because they are neither targets nor features?

 - The variables 'EIN' and 'NAME' were removed as they are neither targets nor features. 

### Compiling, Training, and Evaluating the Model
#### How many neurons, layers, and activation functions did you select for your neural network model, and why?

 - Input Layer: The model’s input layer received all the features after one-hot encoding and scaling.
 - First Hidden Layer: 100 neurons with ReLU activation function.
 - Second Hidden Layer: 30 neurons with ReLU activation function.
 - Third Hidden Layer: 30 neurons with ReLU activation function.
 - Output Layer: 1 neuron with Sigmoid activation function for binary classification.

#### Were you able to achieve the target model performance?

 - Over 200 epochs, we were able to achieve a model with about 80% accuracy at predicting successful campaigns if the following conditions are true:
    - your name appears more than 5 times, and
    - your application type is either: T3, T4, T6, T5, T19, T8, T7, T10, and
    - you have more than 1000 counts in your classification,

#### What steps did you take in your attempts to increase model performance?

Adjusted the number of neurons in each hidden layer.
Experimented with adding and removing hidden layers.
Tried different activation functions in the hidden layers and adjusted the number of epochs.
Processed categorical variables to group rare occurrences into an "Other" category to reduce noise.

## Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.









### Machine Learning Model #1 (Original Data)
#### Balanced Accuracy Score
0.967989851522121

The balanced accuracy score of 0.97 for the original data suggests that the model is able to accurately classify obervations. 

#### Confusion Matrix
array([[18655,   110],
       [   36,   583]])

The original data shows 18,655 healthy loans predicted and 583 unhealthy loans predicted. False positives predicted = 110 and false negatives predicted = 36.

#### Classification Matrix
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.94      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384

The original data shows a high rate of predicting low-risk loans (recall value = 0.99) than high-risk loans (recall value = 0.94). The model  does predicts unhealthy loans with 84% accuracy and predicts healty loan values  with 100% accuracy.

### Machine Learning Model #2 (Re-Sampled Data)
#### Balanced Accuracy Score
0.9935981855334257

The balanced accuracy score of 0.99 for the re-sampled data suggests that the re-sampled model is able to very accurately classify obervations. 

#### Confusion Matrix
array([[18646,   119],
       [    4,   615]])

The re-sampled data shows 18,646 healthy loans predicted and 615 unhealthy loans predicted. False positives predicted = 119 and false negatives predicted = 4.

#### Classification Matrix
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

The re-sampled data shows a very high rate of predicting both low-risk loans (recall value = 0.99) and high-risk loans (recall value = 0.99). The re-sampled model still predicts unhealthy loans with 84% accuracy and predicts healty loan values  with 100% accuracy.

## Summary
Based on the analysis results of the two learning models, the re-sampled (second) model is best for use. The re-sampled model has improved recall values (94% original vs 99% re-sampled) indicating improved high-risk loan prediction rates as well as a significantly smaller amount of false positives identified (36 original vs 4 re-sampled). This model therefore would be extremly useful in predictions loans that could default, and would be therefule useful and reliable for the bank to avoid such high-risk loans.

