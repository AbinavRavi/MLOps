# MLOps
Project to do retraining of the model

## Plan

### Step 1

MNIST model on the mac M1 chip and check how the training is measure performance and run a torchscript to convert the model

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

