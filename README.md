# MLOps
Project to do retraining of the model

## Plan

### Step 1 - Training model and torchscript conversion and model registration

**Task at hand**: Sentiment Analysis

**Dataset used**: [Twitter tweets data](https://www.kaggle.com/datasets/kazanova/sentiment140)

**Skills to be learnt**: 
- Sentiment analysis modelling using pytorch lightning
- Early stopping and tests to verify if model metrics are satisfying (for automatic retraining )
- MLFlow for experiment tracking and hyper parameter search 
- MLFlow model registration and versioning


## Step 2

Create an API for inference and expose it to SVHN Images and check how the inference is

## Step 3

Expectation is that prediction will be wrong

## Step 4

Add a monitoring layer which can monitor if the prediction is wrong

## Step 5 

Monitoring layer should automatically trigger a retraining pipeline with new data included as well (SVHN but single channel)

## Step 6

Automatic stopping and torchscript conversion of model

## Step 7 

A/B test implementation to check if the model is progressing or not

## Step 8

If the model is good then deploy the new model else store metrics and discard

