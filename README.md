# LEAGUE OF LEGEND MATCH PREDICTION

===

Personal Project        
By: Oliver Ton     |     April 2022
 
===
 
Table of Contents
---
 
* I. [Project Overview](#i-project-overview)<br>
[1. Goals](#1-goal)<br>
[2. Description](#2-description)<br>
[3. Initial Questions](#3initial-questions)<br>
[4. Formulating Hypotheses](#4-formulating-hypotheses)<br>
[5. Deliverables](#5-deliverables)<br>
* II. [Project Data Context](#ii-project-data-context)<br>
[1. Data Dictionary](#1-data-dictionary)<br>
* III. [Project Plan - Data Science Pipeline](#iii-project-plan---using-the-data-science-pipeline)<br>
[1. Project Planning](#1-plan)<br>
[2. Data Acquisition](#2-acquire)<br>
[3. Data Preparation](#3-prepare)<br>
[4. Data Exploration](#4explore)<br>
[5. Modeling & Evaluation](#5-model--evaluate)<br>
[6. Product Delivery](#6-delivery)<br>
* IV. [Project Modules](#iv-project-modules)<br>
* V. [Project Reproduction](#v-project-reproduction)<br>
* VI. [KeyTakeaway](#vi-key-takeaway)<br>
 
 
 
## I. PROJECT OVERVIEW
 
 
#### 1.  GOAL:
The goal of this project is to create a model to predict the outcome of a League of Legends match based upon player performance in prior games. Due to the tremendous amount of data available, I anticipate that a model trained on a sufficiently large dataset can achieve a high accuracy at this prediction task.
 
#### 2. DESCRIPTION:
League of Legends is one of the most popular games ever existed, and has one of the most important competitive games. In this game, each team (blue and red) fight to take out the enemy's nexus to win the game. Like all strategy games, we have different objectives in the game which give a certain advantage in the game. This information about these objectives is what will help us make our winning prediction model. To increase our chances to win a game, there are lots of different objectives and events to do, fights to win to increase the power of your champion, as well as winning map terrain by taking down turrets and putting vision on the map.
 
#### 3.INITIAL QUESTIONS:

 
##### Data-Focused Questions

- Which features are correlated with the target variable (hasWon)?
- Does killed or lost a baron nashor or rift herald increase or decrease the chance of winning?
 
##### Overall Project-Focused Questions
- What will the end product look like?
    + A GitHub Repository containing an end-to-end data science pipeline project of personal choosing.
- What format will it be in?
    + A GitHub Repository with a .gitignore, README.md, wrangle.py, work.ipynb, report.ipynb.  
- How will I know I'm done?
   + When I have a question, answer thorugh statistics/visualization, and modeling.
- What is my MVP?
   + A project that runs thorugh the entire pipeline to include modeling, while asking and answering one question. 
- How will I know it's good enough?
   + Has least drop off from train to validate to unseen the model note overfitting. 
 

#### 4. FORMULATING HYPOTHESES

 Null hypothesis: Our target variable not independent to the selected features

 Alternative hypothesis: Our target variable dependent to the selected features
 
#### 5. DELIVERABLES:
- [x] README file - provides an overview of the project and steps for project reproduction
- [x] Draft Jupyter Notebook - provides all steps taken to produce the project
- [x] wrangle.py - provides reproducible code to automate acquiring, preparing, and splitting the data
- [x] Report Jupyter Notebook - provides final presentation-ready wrangle and exploration

 
 
## II. PROJECT DATA CONTEXT
 
#### 1. DATA DICTIONARY:
The final DataFrame used to explore the data for this project contains the following variables (columns).  The variables, along with their data types, are defined below:
 
 
|  Variables             |    Definition                                               |    DataType             |
| :--------------------: | :---------------------------------------------------:       | :--------------------:  |
| gameId                 | Riot API unique game identifier                             |         int64           |
| gameDuration           | Game duration in milliseconds                               |         int64           |
| hasWon                 | If blue team has won the game or not                        |         int64           |
| frame                  | Time in the game in minutes (min)                           |         int64           |
| goldDiff               | Blue team gold difference                                   |         int64           |
| expDiff                | Blue team experience difference                             |         int64           |
| champLevelDiff         | Blue team champions level difference                        |         float64         |
| isFirstTower           | If the blue team destroyed the first tower (gold bonus)     |         int64           |
| isFirstBlood           | If the blue team killed the first enemy (gold bonus)        |         int64           |
| killedFireDrake        | Number of fire dragons killed by the blue team              |         int64           |
| killedWaterDrake       | Number of water dragons killed by the blue team             |         int64           |
| killedAirDrake         | Number of air dragons killed by the blue team               |         int64           |
| killedEarthDrake       | Number of earth dragons killed by the blue team             |         int64           |
| killedElderDrake       | Number of elder dragons killed by the blue team             |         int64           |
| lostFireDrake          | Number of fire dragons killed by the red team               |         int64           |
| lostWaterDrake         | Number of water dragons killed by the red team              |         int64           |
| lostAirDrake           | Number of air dragons killed by the red team                |         int64           |
| lostEarthDrake         | Number of earth dragons killed by the red team              |         int64           |
| lostElderDrake         | Number of elder dragons killed by the red team              |         int64           |
| killedBaronNashor      | Number of Barons Nashor killed by the blue team             |         int64           |
| lostBaronNashor        | Number of Barons Nashor killed by the red team              |         int64           |
| killedRiftHerald       | Number of Rift Heralds killed by the blue team              |         int64           |
| lostRiftHerald         | Number of Rift Heralds killed by the red team               |         int64           |
| kills                  | Blue team total kills                                       |         int64           |
| deaths                 | Blue team total deaths                                      |         int64           |
| assists                | Blue team total assists                                     |         int64           |
| wardsPlaced            | Blue team total wards placed                                |         int64           |
| wardsDestroyed         | Blue team total enemy wards destroyed                       |         int64           |
| wardsLost              | Blue team total wards destroyed by the red team             |         int64           |

## III. PROJECT PLAN - USING THE DATA SCIENCE PIPELINE:
The following outlines the process taken through the Data Science Pipeline to complete this project. 
 
Plan➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver
 
#### 1. PLAN
- [x]  Review project expectations
- [x]  Draft project goal to include measures of success
- [x]  Create questions related to the project
- [x]  Create questions related to the data
- [x]  Create a plan for completing the project using the data science pipeline
- [x]  Create a data dictionary to define variables and data context
- [x]  Draft starting hypothesis
 
#### 2. ACQUIRE
- [x]  Create .gitignore
- [x]  Create wrangle.py module
- [x]  Store functions needed to acquire the dataset from kaggle
- [x]  Ensure all imports needed to run the functions are inside the wrangle.py document
- [x]  Using Jupyter Notebook
     - [x] Run all required imports
     - [x] Import functions from wrangle.py module
     - [x] Summarize dataset using methods and document observations
 
#### 3. PREPARE
Using Jupyter Notebook
- [x]  Import functions from wrangle.py module
- [x]  Summarize dataset using methods and document observations
- [x]  Clean data
- [x]  Features need to be turned into numbers
- [x]  Categorical features or discrete features need to be numbers that represent those categories
- [x]  Address missing values, data errors, unnecessary data, renaming
- [x]  Split data into train, validate, and test samples
Using Python Scripting Program (Jupyter Notebook)
- [x]  Create prepare function within wrangle.py
- [x]  Store functions needed to prepare the data such as:
   - [x]  Cleaning Function: to clean data for exploration
- [x]  Ensure all imports needed to run the functions are inside the wrangle.py document
 
#### 4.EXPLORE
Using Jupyter Notebook:
- [x]  Answer key questions about hypotheses
     - [x]  Run at least two statistical tests
     - [x]  Document findings
- [x]  Create visualizations with the intent to discover variable relationships
     - [x]  Identify variables related to hasWon
     - [x]  Identify any potential data integrity issues
- [x]  Summarize conclusions, provide clear answers, and summarize takeaways
     - [x] Explain plan of action as deduced from work to this point
 
#### 5. MODEL & EVALUATE
- [x] Modeling was necessary for this project to see if the features is the best to predict the match outcome. Select the best model to test on unseen data

#### 6. DELIVERY
- [x]  Prepare final notebook in Jupyter Notebook
     - [x]  Create clear walk-though of the Data Science Pipeline using headings and dividers
     - [x]  Explicitly define questions asked during the initial analysis
     - [x]  Visualize relationships
     - [x]  Document takeaways
     - [x]  Comment code thoroughly

## IV. PROJECT MODULES:
- [x] wrangle.py - provides reproducible python code to automate acquiring, preparing, and splitting the data
- [x] explore.py - provides reproducible python code to automate plots and statistical test
- [x] model.py - provides reproducible python code to automate models (decision tree, knn, random forest), create baseline
   
## V. PROJECT REPRODUCTION:
### Steps to Reproduce
- [x] Make .gitignore and confirm .gitignore is hiding your env.py file
- [x] Clone the repo (including the wrangle.py, explore.py, model.py)
- [x] Import python libraries:  pandas, matplotlib, seaborn, numpy, scipy, and sklearn
- [x] Follow steps as outlined in the README.md
- [x] Run Final_Report.ipynb to view the final product

## VI.KEY TAKEAWAY:
### Conclusion
This individual project aimed to identify predictors of match wins for the game Leauge of Legend. The matches outcome data was from kaggle and explored using visualization and statistical tests, to identify which features to use in model creation. I then applied models, including Decision Tree, KNN, and Random Forest and was able to beat baseline predictions by 11.56%.

### Recommendations
- From modeling, it suggested that feautres like get first turret, first blood, killed drake, elder drake and baron nashor. These features were the best at predicting match otucome.

### Next Steps
- With more time, I would like to use rfe or kbest as features selection for my models, it might be able to improve model accuracy. And look at other features like kills, death, assist those can be predict the match outcome.
