# Audio emotion classification
Repository concerning Summer 2022 project of exam data science laboratory at Politecnico di Torino.

This is just a brief introduction to what I've done. If you want to further inspect the proposed solution, I invite you to check the report. __All of this was done in just 2 weeks__.

# Task
Create a classification model upon audios able to detect 7 different classes and evaluate it using the __F1-macro score__ (__baseline__ to surpass was __0.544__).

Labels are the following:
  - Surprised 
  - Neutral
  - Disgusted 
  - Fearful
  - Sad
  - Angry 
  - Happy
  
# Dataset
The given dataset was split in 2 parts:
- development.csv: this file contains the ids of the labeled audios as well as their labels.
- evaluation.csv: this file contains just the ids of the audios used to evaluate proposed models

A folder containing all audio files were provided too.

An inspection of development dataset showed that classes were imbalanced. Hence, oversampling was used.

![class imbalance](https://github.com/notlosca/audio_emotion_classification/blob/main/images/num_samples_per_class.svg)

# Features 
Features used are the following:
1. Mel-Frequency Cepstrals Coefficients (MFCCs)
2. delta MFCCs
3. delta-delta (order 2 delta) MFCCs
4. Chromagram
5. Time duration
6. Zero Crossing Rate
7. Root Mean Square energy

A dimensionality reduction was applied to select the most relevant features.

# Classifiers
In performance order:
|Classifier|F1-macro|
|----------|--------|
|SVM| __0.707__| 
|Voting Classifier (soft)| 0.701|
|Random Forest|0.668|

All proposed classifiers were trained using 10-fold cross-validation. Grid search was used to tune hyperparametes.

# Conclusion
The goal to surpass the naive baseline of 0.544 was achieved by proposed classifiers.
There are many feasible scenarios that could further improve results. For example the usage of neural networks which has been proven to work pretty well. Such architectures were not used consciously since are out of scope of this exam.
