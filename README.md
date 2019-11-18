## Introduction and Motivation
Parkinson’s disease (PD) is the most common neuro-degenerative movement disorder [1]. Characterized by the degradation of dopaminergic neurons, PD patients develop severe motor symptoms and cognitive impairment. Unfortunately, treatment options for PD symptoms do not reverse the underlying neuronal degradation and disease progression [1]. Currently, the best method of defense against PD is early detection. However, patients are often diagnosed with PD by severe motor dysfunction, occurring around 80% degeneration of dopamine neurons [2]. Therefore, it is important to diagnose PD before its advancement to preserve neuron integrity and slow progression.

One strategy for early detection of PD is speech pattern recognition. PD vocal dysfunction may be identified 5 years before traditional diagnoses by changes including, reduced volume and tongue flexibility, narrow pitch range, and long pauses [3]. In this project, we used various speech-related data sets to classify and predict PD disease severity.

## Data Description
The aim of this project is to predict Parkinson's presence, and disease severity. The Unified Parkinson Disease Rating Scale (UPDRS) is a rating tool used to understand the course of PD in patients. The UPDRS scale is used for determining treatment strategies and for PD clinical research purposes. The scale includes ratings for PD symptoms and is comprised of 6 parts:
1. Mentation, Behavior, and Mood
2. Activities of Daily Living
3. Motor Examination
4. Complications of Therapy
5. Modified Hoehn and Yahr Scale
6. Schwab and England Activities of Daily Living Scale

UPDRS scores range from 0 to 199, with 199 representing the greatest disease severity, or total disability [4].

We include two datasets in our project as the training data. We use MSR or the combination of TE and MSR for predicting Parkinsons's presence, and we use TE or the MSR/TE combination for predicting disease severity (UPDRS).

### [Multiple Sound Recording (MSR) Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings) [5]

##### Training data
The training data were gathered from 20 patients with Parkinsons and 20 healthy individuals. Multiple types of sound recordings were taken from each participant (listed below) and expert physicians assigned each participant a UPDRS score.

**Utterances**
* 1: sustained vowel (“a”)
* 2: sustained vowel (“o”)
* 3: sustained vowel (“u”)
* 4-13: numbers from 1 to 10
* 14-17: short sentences
* 18-26: words

**Features Training Data File:**
* column 1: Subject id
* columns 2-27: features

* features 1-5: Jitter (local), Jitter (local, absolute), Jitter (rap), Jitter (ppq5), Jitter (ddp) - Several measures of variation in fundamental frequency
* features 6-11: Shimmer (local), Shimmer (local, dB), Shimmer (apq3), Shimmer (apq5), Shimmer (apq11), Shimmer (dda) - Several measures of variation in amplitude
* features 12-14: AC, NTH, HTN - Autocorrelation and ratio of noise to tonal components in the voice
* features 15-19: Median pitch, Mean pitch, Standard deviation, Minimum pitch, Maximum pitch
* features 20-23: Number of pulses, Number of periods, Mean period, Standard deviation of period
* features 24-26: Fraction of locally unvoiced frames, Number of voice breaks, Degree of voice breaks
* column 28: UPDRS
* column 29: Disease Class

##### Testing data
The testing data were gathered from 28 different patients with Parkinsons. The patients are asked to say only the sustained vowels 'a' and 'o' three times each, producing 168 recordings. The same 26 features are extracted from the voice samples.

**Utterances**
* 1-3: sustained vowel (“a”)
* 4-6: sustained vowel (“o”)

### [Telemonitoring (TE) Dataset](http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) [6]
The data was gathered from 42 people with early-stage Parkinson's disease. There are 16 voice measures, and two regression measurements: motor UPDRS and total UPDRS. Each row of the dataset contain corresponds to one voice recording. There are around 200 recordings per patient, the subject number of the patient is identified in the first column.

**Features**
* subject# - Integer that uniquely identifies each subject
* age - Subject age
* sex - Subject gender '0' - male, '1' - female
* test_time - Time since recruitment into the trial. The integer part is the number of days since recruitment.
* motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated
* total_UPDRS - Clinician's total UPDRS score, linearly interpolated
* Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP - Several measures of variation in fundamental frequency
* Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA - Several measures of variation in amplitude
* NHR, HNR - Two measures of ratio of noise to tonal components in the voice
* RPDE - A nonlinear dynamical complexity measure
* DFA - Signal fractal scaling exponent
* PPE - A nonlinear measure of fundamental frequency variation

### Dataset Comparison

Dataset | Data Points | Features
------------ | ------------- | -------------
Multiple Sound Recoring Training | 1040 | 29
Multiple Sound Recoring Testing | 168 | 28
Telemonitoring | 5875 | 22

We observed that the TE dataset has over 5 times as many data points as the MSR data sets. In our experiment, we use MSR and TE for training both classification and regression models.

### Response data: UPDRS
Although TE and MSR have different UPDRS score distributions, their score range is similar. As a result, we decided to try merging these two datasets together by their common features to build another, larger training dataset.
Distributions of UPDRS scores:

<img src="./visualizations/UPDRS_TE_MSR.jpg">

## Intial Data Exploration & Unsupervised Learning

### Covariance Matrices

First began by creating visualizations for the correlation matrix of each dataset. We used a special heatmap that encodes correlation using not only color, but size. The code for this visualization was taken from this [tutorial](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec).

#### Telemonitoring
<img src="./visualizations/Telemonitoring-correlation-matrix.jpg" width="700" height="700">

#### Multiple Sound Recording
These datasets have two separate train and test datasets. The data are drawn from very different distributions of people: the training dataset comes from a mix of people with and without Parkinson's, and the testing dataset is entirely composed of data collected from people with Parkinson's. We graphed the covariance matrix of both below.

##### Training Data Set

<img src="./visualizations/multiple-sound-recoring-train-correlation-matrix.jpg" width="700" height="700">

###### Testing Data Set
<img src="./visualizations/multiple-sound-recoring-test-correlation-matrix.jpg" width="700" height="700">

### Dimension Reduction:
Each dataset has many features, and from the covariance matrices we see that some features are highly correlated, so we tried some dimensionality reduction using principal component analysis (PCA) and linear discriminant analysis (LDA).

Below are results we obtained by using PCA on the datasets and seeing how many components we need to recover 99% of the variance.

#### Principal Component Analysis

Dataset | # of Features | Reduced # of Components
------------ | ------------- | -------------
Multiple Sound Recoring Training | 29 | 17
Multiple Sound Recoring Testing | 28 | 12
Telemonitoring | 22 | 11

<img src="./visualizations/PCAVariance.png" width="400" height="250">

We successfully reduced the number of components to about half, while still recovering 99% of the variance in these datasets.

We also created visualizations for the first two components of the TE and MSR train dataset. We colored datapoints by the UPDRS score, a measure of PD severity. For LDA, UPDRS scores were rounded, binned, and used for classes.
<img src="./visualizations/PCA_LDA_TE_MSRtrain.jpg">

Patients of varying disease severity were not separated in PCA plots. We saw some separation of patient recordings with the highest and lowest UPDRS scores in the TE dataset accross component 2 on the y-axis.

## Supervised Learning & Prediciton of PD
### Datasets
For this section, we explored the prediction of PD in the MSR testing set from 1) the MSR training set and 2) an aggregated TE and MSR training data set.

For the latter, we found 13 overlapping features: Jitter, Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP, Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA, NHR, and HNR. We reduced the MSR training set and TE datasets to these features and performed predictions.

### Classification: PD and non-PD patients

At this point, we used the MSR training set and the aggregated MSR/TE set to train binary classification models and tested them on the MSR testing set. In the testing set, all of the samples have PD. Our goal is to identify the accuracy of PD diagnoses among these patients. Or, in other words, see how many of the 28 patients' recorings our model can successfully diagnose with PD.

We also compare performances accross different algorithms. As shown in the following table, Naive Bayes can achieve the best result when using the MSR training set. Also, after combining both sets, our accuracy improved for each classifier. However, there is also a concern here that samples with and without disease are very imbalanced in the combined set. Adding in the TE set with all PD patients, may skew our models to predict PD, the only class of datapoint in the testing set, more frequently. The improvement might be caused by overfitting. Moreover, from these experiments, we do not know how our model will perform for datasets of non-PD patients, a significant limitation.

#### Accuracy of each classifier:

Classifier | MSR Train | MSR Train & TE
------------ | ------------- | -------------
Random Forest Classifier | 57.1% | 100%
Decision Tree | 61.3% | 80.4%
Naïve Bayes | 84.5% | 86.9%
Neural Network | 65.5% | 100%

#### SVM Classifier:
We investigated the use of several kernels of increasing complexity: a basic linear kernel, assuming our data are linear separable; a polynomial (poly) kernel, assuming our data are nonlinear; and a more complex radial basis kernel (RBF), with a feature space of infinite dimension.

Kernel | MSR Train | MSR Train & TE
------------ | ------------- | -------------
Linear | 51.7% | 100%
Poly | 79.8% | 96.4%
RBF | 54.1% | 99.4%

Here, we see that the polynomial kernel is the most accurate for the MSR training data and is comparable to the accuracy of the Naïve Bayes classifier. We also see that each kernel in the MSR/TE combined dataset has high accuracy. Again, the high accuracy in the MSR/TE combined set may be due to the problems mentioned above.

### Regression: UPDRS scores

Since we have UPDRS scores for the MSR training dataset and the TE dataset, we wanted to test if we could predict these scores with different regression models. We first explore the TE dataset for training our models and then use the TE and MSR training combined set. This time, the combined data set will be normalized (l2 norm) before using the supervised learning algorithms. For both, we only choose patients with Parkinson disease for our training data. To evaluate our result, we utilize the evaluation method proposed by the original paper, which split 10% and 90% data from TE dataset for 10 fold cross-validation and report the Mean Absolute Error (MAE) at the testing subset.

#### Training on TE

We report the best result proposed by the original paper, which used Jitter(Abs), :Shimmer, NHR, HNR,
DFA, and PPE as the training features and Classification And Regression Tree (CART) as the model.
In our experiment, we leveraged different feature processing methods for pre-processing our training data. We compare three methods 1) using original features 2) log transformation of the features and 3) PCA feature reduction. The results are shown in the following table. We also tried various regression models and report the best two models (Random Forest Regressor and Extra Trees Regressor) here. By using Extra Trees Regressor and PCA dimension reduction, we can achieve an MAE score 4.38, which shows PCA has the ability to effectively improve the accuracy of these regressors and capture feature correlation.

Method | feature | MAE
------------ | ------------- | -------------
Best result proposed by paper | log transform | 7.52±0.25
Random Forest Regressor | original data | 6.46
Extra Trees Regressor | original data | 6.41
Random Forest Regressor | log transform | 6.42
Random Forest Regressor | PCA | 4.81
Extra Trees Regressor | PCA | 4.38

#### Training on TE & MSR combination

We also try to train our regression model with the aggregated set.

Method | Features | MAE
------------ | ------------- | -------------
Random Forest Regressor | original data | 8.56
Extra Trees Regressor | original data | 8.3
Random Forest Regressor | PCA | 8.08
Extra Trees Regressor | PCA | 8.16

While the accuracies of the models trained on the joint dataset are worse, this is almost certainly due to the inherent differences in the two datasets. The TE dataset has over 100s of samples from each individual over time, while the MSR dataset has significantly less. Furthermore, the MSR dataset has participants say a preselected set of phrases, while the TE dataset is a collection of varying and unscripted phrases from participants. It is possible that this mix of data leads to a more generalized, better model, but more experiment needs to be done to determine this.

#### Neural Network Regression

We also decided to try building some neural network models to test their efficacy. Due to the relatively low feature space and number of data points, we assumed they may be less effective than some other classical methods, but they ended up being just as effective, if tuned properly.

We initially assumed that the neural network should be a single hidden layer deep, to handle a lack of linear seperability but to also prevent overfitting. We tried many different hidden layer sizes, with the joint dataset and PCA preprocessing. The results are shown below:

Hidden Layer Size | MAE
------------ | -------------
10 | 8.71
20 | 8.64
50 | 8.66
100 | 8.43
200 | 8.34
500 | 8.36
1000 | 8.36

Since none of these networks beat our prior results, we decided to try neural networks with two hidden layers. The results were very impressive!

Hidden Layer 1 Size | Hidden Layer 2 Size | MAE
------------ | ------------- | -------------
100 | 50 | 8.36
200 | 100 | 8.31
500 | 200 | 8.22
1000 | 500 | 8.07

Running the two-layer networks on the TE dataset, we got even more impressive results:

Hidden Layer 1 Size | Hidden Layer 2 Size | MAE
------------ | ------------- | -------------
100 | 50 | 3.27
200 | 100 | 3.31
500 | 200 | 3.24
1000 | 500 | 3.14

In the future, it would be interesting to do explore why neural networks with two layers and many nodes result in lower error on these datasets.

## Applying our Models to Available Data
One of the larger open datasets is the Mozilla Foundation's [Common Voice](voice.mozilla.org) dataset. It has 2454 hours of sound recordings spanning 29 languages. The data is published under C-0 license, which means all data is public domain.

To reduce the data-points processed, the models were only tested on validated American-English recordings. Furthermore, since Parkinson's symptoms typically first manifest around the age of 50, all recordings from people ages 39 and younger were filtered out, to allow for some leeway. This left us with 46,813 sentence-long sound files.

### Dataset Composition
Total Samples: 46,813

Age | Number of Samples
------------ | -------------
40's | 21,824
50's  | 10,700
60's | 12,299
70's  | 1,661
80's  | 276
90's  | 53

### Results
The tables below give the percent of Mozilla Dataset Samples predicted to have Parkinsons. The population rate is ~1%.

#### Classifier Predictions:

Classifier | MSR Train | MSR Train & TE
------------ | ------------- | -----
Random Forest Classifier | 7.7% | 100%
Decision Tree | 48.5% | 61.1%
Naïve Bayes |  16.5% | 49.1%
Neural Network |  16.8% | 100%

#### SVM Predictions by Kernel:

Kernel | MSR Train | MSR Train & TE
------------ | -------------| -----
Linear |  48.1% | 99%
RBF  | 13.1% | 73.8%
Poly | 24.8% | 94.3%


### Discussion
It is clear that there is over-prediction when it comes to the real world data, especially in the case of MSR + TE. But, the MSR + TE results can be explained by two factors:
1. The datasets had a limited overlap in features, meaning that the featureset used was not as rich as was possible from using either individually.
2. The TE dataset is composed entirely of people with early stage Parkinson's. This means that the data available was imbalanced, leading to overfitting even if cross-validation methods are used.

Without this imbalance, examining the MSR data-set alone provided much more realistic results. The best performing methods were Random Forest Classifier, Naïve Bayes, and the RBF SVM. All of these tell us that any separation requires an accounting for the non-linear relationship between the parameters.

What we can draw from this analysis is that that overlapping set of features, which are Jitter, Shimmer and Harmonicity are not sufficient to detect parkinsons. But, while information on voice breaks, pitch, and autocorrelation do provide additional predictive power, that too is not entirely sufficient for building a model for Parkinson's classification.

### Conclusion & Limitations
In this project, we leverage several unsupervised (PCA, LDA, etc.) and supervised (SVM, Naïve Bayes, etc) techniques for predicting Parkinson disease based on speech analysis data. Our methods not only achieve high accuracy on predicting Parkinson's presence, but also gain a significant improvement on predicting UPDRS score.

However, the biggest limitation of our methods is the robustness. Since all of our models are based on two dataset (TE and MSR), when the testing data is gathered from different environment setting, such as longer samples, various pitch range or other data variance, then our models' performance will be easily influenced by these factors.

### Future Work
This project leaves open several avenues for further research. Some of those are listed below:

1. Use the Disease Classification Dataset to create models that take advantage of other speech-derived features, such as Mels Frequency Ceptral Coefficients (MFCC's) and tunable-Q factor wavelet transform (TQFWT's). These features may lead to more accurate regression and classification.
2. The Michael J Fox Foundation has a [dataset](https://www.kaggle.com/c/predicting-parkinson-s-disease-progression-with-smartphone-data/data) composed of smart phone data collected from 9 PD patients at varying stages of the disease and 7 control volunteers, roughly matched for age and gender over the course of 4 months. It includes raw audio and accelerometry data, which could both be utilized to create better models.
3. Using the raw audio with deep neural networks would be another interesting area for exploration, as all of these datasets are based on speech-dervied features. The Michael J Fox dataset would be a good starting point for this research.

### Significance
An estimated 10 million people worldwide have PD. Since there is no effective cure for PD, early detection is the primary goal. However, PD is difficult to diagnose because symptoms often manifest in the late stages of the disesase. Therefore, there is a great need for early detection and monitoring methods, like speech analysis.

In this project, we classified and predicted PD severity on the UPDRS. Our goal was to achieve accuracy that is comparable to average clinical diagnosis accuracy. A meta-analysis from 2016 of 20 studies showed that diagnostic accuracy was around 80% [7]. While we acknowledge the limitations mentioned above, we were able to reach this threshold with the TE/MSR datasets.

Overall, speech detection of PD can be a powerful tool in early detection and monitoring of disease severity. We believe more effort should be invested in this field.

### Contribution
Amy Hung - Classification, Regression

Anthony D'Achille - Data exploration, Unsupervised Learning, Regression

Chris Scherban - Real life data feature extraction and prediction

Jess Hoffman - Data aggregation, Regression

Yiting Zhang - Classification, Regression


## References
[1] Rascol, O.,Payoux, P.,Ory, F.,Ferreira, J. J., Brefel-Courbon, C. and Montastruc, J. (2003), Limitations of current Parkinson's disease therapy. Ann Neurol., 53: S3-S15.

[2] Pagan, F. L., (2012). Improving outcomes through early diagnosis of Parkinson’s disease. The American Journal of Managed Care, 18, 176-182.

[3] Vaiciukynas, E., Verikas, A., Gelzinis, A., & Bacauskiene, M. (2017). Detecting Parkinson's disease from sustained phonation and speech signals. PloS one, 12(10), e0185613. doi:10.1371/journal.pone.0185613

[4] UPDRS SCALE [Internet]. Theracycle. PD Resources; [cited 2019Nov16]. Available from: https://www.theracycle.com/pd-resources/links-and-additional-resources/updrs-scale/

[5] Erdogdu Sakar, B., Isenkul, M., Sakar, C.O., Sertbas, A., Gurgen, F., Delil, S., Apaydin, H., Kursun, O., 'Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings', IEEE Journal of Biomedical and Health Informatics, vol. 17(4), pp. 828-834, 2013.

[6] Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2009). Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests. IEEE transactions on Biomedical Engineering, 57(4), 884-893.

[7] Rizzo G et al. Accuracy of clinical diagnosis of Parkinson disease: A systematic review and meta-analysis. Neurology 2016 Jan 13; [e-pub]. (http://dx.doi.org/10.1212/WNL.0000000000002350)
