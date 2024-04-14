Facial Keypoints Detection
===========================

<img align="right" width="180" src="./images/cu_logo.jpg"/>

### Final Project for W4732 Computer Vision II - Spring 2024
#### Author : [Cristopher Benge](https://cbenge509.github.io/) (cb3704)

<br><br><br><br>
Columbia University in the City of New York

Masters of Science in Computer Science - Machine Learning Concentration <br>
Section V01 - [Carl Vondrick, PhD](https://www.cs.columbia.edu/~vondrick/)

---

<img src="./images/seth_logo.png" align="center">

---

## Description

This repo contains a solution for the [Kaggle Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) challenge, developed by Cristopher Benge.  This solution leverages a variety of neural network architectures, image augmentation techniques, and a custom K-fold generalized stacker to achieve a top 5 score on the Kaggle private leaderboard.

#### Highlight of key files included in this repository:

  |File | Description |
  |:----|:------------|
  |[Facial Keypoints - EDA.ipynb](Facial%20Keypoints%20-%20EDA.ipynb) | Jupyter Notebook containing exploratory data analysis findings of the competition data.|
  |[prepare_data.py](prepare_data.py) | Performs preprocessing of TRAIN and TEST datasets.|
  |[train_model.py](train_model.py) | Trains one or more models based on user-provided arguments.|
  |[predict.py](predict.py) | Predict from last trained version of one or more user-specified models.|
  |[train_stack.py](train_stack.py) | Performs K-Fold generalized stacking using the user-specified models.|
  |[predict_stack.py](train_stack.py) | Performs metaregressor traning and prediction from all stacked model files.|
  |[generate_submission.py](prepare_data.py) | Generates submissions of individual models by combining the 8-only and Not-8-ony predictions.|
  |[data_loader.py](/utils/data_loader.py) | Utility class used for loading the competition dataframes.|
  |[data_transformer.py](/utils/data_transformer.py) | Utility class for performing data cleaning and augmentation.|
  |[model_zoo.py](utils/model_zoo.py) | Utility class; provides model construction from the model zoo construction and base prediction routines.|

---

## Performance

This solution achieves a best score of **1.28637 RMSE**, which is good for a 2nd place position on the [now locked] private leaderboard (and within 0.004 RMSE of the 1st place position):

<img src="./images/final_submission.JPG" width="800"/>
<img src="./images/top10.jpg" width="400"/>

---

## Evaluation Criteria

The Kaggle facial keypoints detection challenge asks competitors to predict, with as small an RMSE as possible, the (x,y) point coordinates for [up to] 15 facial landmarks (left center eye, nose tip, left outer eyebrow, etc) per image on 1,783 images in the provided TEST dataset.  All images are sized as 96x96x1 grayscale.  Each point is submitted as their individual X and Y coordinate values for separate scoring, thus competitors will need to submit up to 15 X and Y value pairs for an image containing all 15 landmarks.  Not all images contain 15 landmarks present.

The competition is ranked by the lowest error in prediction by the [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation):

<img src="./images/rmse.jpg" width="400" />

---

## Training Pipeline

The training pipeline begins with applying cleaning and augmentation to all images, and then splitting the TRAIN and TEST datasets into two separate challenges (more on the reason for this below).  I experiemented with a lot of augmentation strategies, but in the end the following performed best for most models evaluated:

* Manual label re-assignment to 56 training labels that were overtly incorrect in the TRAIN set.
* Image pixel values normalized between 0 and 1 (both TRAIN and TEST).
* Positive and negative rotation of images by the following degrees: {+/-5, +/-9, +/-13}
* Elastic transformation of images using alpha/sigma values of {991/8, 1201/11}.
* Random gaussian noise, applying between ~-0.03 to ~+0.03 values to the pixels.
* Brightened images by the following scalars: {1.2x, 1.5x}.
* Dimmed images by the following scalers: {0.6x, 0.8x}.
* Contrast stretching in the following ranges: {0.0 - 1.5, 0.0 - 2.0}.
* Custom image sharpening 3x3 conv2d kernel (applied one time).
* Contrast Limited AHE (i.e. [adaptive histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)) used with clipping of 0.03.
* Horizontal flipping / mirroring.

Many other augmentations were experimented with, such as horizontal blurring, laplacian smoothing, etc. but all resulted in adverse impact to prediction error and were eliminated after empirical testing.

The TRAIN dataset provided by the competition organizers appears to be derived from two distinct sources: one where all 15 landmark labels are present, and another where only 4 are present (note: there are a few images thorughout that have some other count of missing label information, but they are the exception).  Initially, I was able to achieve a single best-model RMSE score of **1.43812** by addressing all missing labels through interpoloation of their `np.nanmean()` keypoint average location.

![train pipeline](/images/train_pipeline.png)

I discovered significant improvement (approx. _-0.2 RMSE_) through splitting the training pipeline in two: (1) trained models on the set with all data points available, and (2) trained models for only the 4 keypoints present in the "partial" set.  At prediction time, I use the "partial" models to predict only those TEST records for which we are asked to predict 4 points, and use the "complete" (1) models for all other facial keypoints.  Controlling for all other changes, this move alone resulted in a best single-model score of RMSE **1.35183**.

Generalized stacking using a K-Fold (K=5) approach was used to avoid overfitting at the metaregressor phase.  Ten neural networks were used as Level 1 predictors for both the "complete" and "partial" data sets (the models were identical save for the final layer).  All models were trained using a batch size of 128 and epoch size of 300.  Training occurred on a local machine with 1 x NVidia RTX 4090.  A general descripton for each Level 1 model is listed below:

| Model Name | Description |
|:-----------|:------------|
|Conv2D 5-Layer | A simple 5-layer 2D convolutional neural network.  |
|NaimishNet | A 7.4M parameter 2D convolutional network that learns only one keypoint at a time ([paper](https://arxiv.org/abs/1710.00977)).  |
|Conv2D 10-Layer | A deeper version of Conv2D 5-Layer above. |
|Local2D | A modified version of Conv2D 5-Layer whose final layer is a Local2D with global average pooling. |
|Inception V1 | A modified version of Google's [inception V1](https://arxiv.org/abs/1409.4842) model. |
|Inception V3 | A modified version of Google's [inception V3](https://arxiv.org/abs/1512.00567) model. |
|LeNet5 | A slightly modified 5-layer version of the [classic LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) model. |
|ResNet50 | A slightly modified version of Microsoft's classic [ResNet50](https://arxiv.org/abs/1512.03385) model. |
|ResNet | A customized, greatly simplified ResNet model architecture. |
|ResNeXt50 | An implentation of the full [ResNeXt50](https://arxiv.org/abs/1611.05431) model. |

Following K-Fold training of the ten models for both the "complete" TRAIN dataset and the "partial" TRAIN dataset, all 10 models are then trained again on the complete TRAIN dataset for the prediction phase, used during final inferencing.  After all L1 training predictions are captured, simple multiplciation feature interactions are generated on a per model basis, resuliting in a per-model input space increase from 30 (the initial 30 keypoint X and Y coordinates) to 30 + (30-choose-2) feature interactions = 465 per model inputs.  A [MultiTaskElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html) linear regression biased to L1 regularization @ 100% is used as our final regressor over all (465 * 7) = 3,255 inputs.  This model is then saved and used in the final inferencing for submission of predictions.

---

## Inference Pipeline

The inferencing pipeline behavior is essentially duplicative of the training process above, with the exception that TEST is also split based on whether we are scored on finding 4 keypoints (8 labels) or NOT 4 keypoints (Not 8 labels).  These images are inferenced through all sevel Level 1 models, combined, feature interactions are calculated, and a final submission inference is taken from the Level 2 MultiTaskElasticNet linear regressor.

![inference pipeline](/images/inference_pipeline.png)

---

## Generalized Stacking

A [Stacked generalization](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) approach was used to leverage the strengths of each model to produce a final prediction.  To avoid overfitting our level-1 regressors, we used K-fold (K=5) cross-validation which added regularization to our final metaregressor.  At each k-th iteration, one k fold is held out to act as the validation set for training.  The predictions from each model on each k-th interval are stored and used at training time for the final metaregressor.  Prior to final metaregressor training, simple multiplication feature interactions are added on each of the 30 (or 8) predictions made by the level-1 regressors.  Finally, the linear metaregressor is fit to all values and used to predict the final submission values. 

![train pipeline](/images/kfold_stacking.png)

---

Models Architectures
=====================

Below are reference model architecture depictions for the ten Level-1 regressors.  Source code for each can be viewed in the ``utils\model_zoo.py`` file.

## Conv2D 5-Layer

<img src="./models/KERAS_CONVNET5/ALL_FEATURES_30_model_plot.png" width="400"/>

## Inception V1

<img src="./models/KERAS_INCEPTION/inception_ALL_FEATURES_30_plot.png" width="800"/>

## Inception V3

<img src="./models/KERAS_INCEPTIONV3/inception_ALL_FEATURES_30_plot.png" width="500"/>

## Conv2D 10-Layer

<img src="./models/KERAS_KAGGLE1/ALL_FEATURES_30_model_plot.png" width="300"/>

## Local2D

<img src="./models/KERAS_KAGGLE2/ALL_FEATURES_30_model_plot.png" width="800" />

## NaimishNet

<img src="./models/KERAS_NAIMISHNET/NaimishNet_left_eye_center_30_model_plot.png" width="800" />

## LeNet5

<img src="./models/KERAS_LENET5/ALL_FEATURES_30_model_plot.png" width="500"/>

## ResNet50

<img src="./models/KERAS_RESNET50/ALL_FEATURES_30_model_plot.png" width="300"/>

## ResNeXt50

<img src="./models/KERAS_RESNEXT50/ALL_FEATURES_30_model_plot.png" width="300"/>

---

License
-------
Licensed under the MIT License. See [LICENSE](LICENSE) file for more details.
