# Module 20 Report Template

## Overview of the Analysis

The purpose of this analysis was to develop a tool for the Alphabet Soup nonprofit foundation that can help it select the applicants for funding with the best chance of success in their ventures. It uses the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. The target predictive accuracy was a minimum of 75%. 

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

 - Yes, the model was able to acheive greater than the target accuracy of 75%. By running 200 epochs, we were able to achieve a model with about 80% accuracy at predicting successful campaigns if the following conditions are true:
    - your name appears more than 5 times, and
    - your application type is either: T3, T4, T6, T5, T19, T8, T7, T10, and
    - you have more than 1000 counts in your classification.
  

#### What steps did you take in your attempts to increase model performance?

 - The steps taken to increase model performance included:
    - Adjusted the dropped non-beneficial columns
    - Processed categorical variables to group rare occurrences into an "Other" category to reduce noise.
    - Adjusted the number of neurons in each hidden layer
    - Experimented with adding and removing hidden layers
    - Tried different activation functions in the hidden layers and adjusted the number of epochs.
    - Adding additional epochs to give model more chances to increase accuracy
    - Compared with Random Forest model to compare accuracy (random forest was only 78% accurate)

## Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

Based on the analysis results, the revised neural network model was able to exceed the target accuracy of 75%, by creating a model with 80% accuracy. Even with a loss of about 45%, this model should be very useful and reliable in predicting successful applicants for the Alphabet Soup nonprofit foundation.

It should be noted that using other models may also be able to provide additional and better insight into the performance and prediction capabilities. A model such as Random Forest, used at the end of the analysis has the advantage of being easier to interpret, they require less data preprocessing, less training time, less prone to overfitting, and more versatile. Using such a model as Random Forest may be able to highlight feature importance giving more insight into how better to predict succesful applicants. 

