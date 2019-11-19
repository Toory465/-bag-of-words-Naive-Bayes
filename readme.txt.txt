This is the explanation for Kaggle competition. 
The implementation includes the implementation for Random classifier, Naive base 
classifier and Naive base classifier with Laplas smoothing

In the implementation there are three phases: 
1.	Data loader:
a.	Load data
b.	Using NLTK to filter words and only keep alphabetical words
c.	Filter words with less than two characters
d.	Convert all words to lower case
e.	Stemming
2.	Naïve base model
a.	Develop the naïve base model 
b.	Develop the naïve base model with laplas smoothing 
i.	I tried different alpha and the best one was alpha = 2.15
3.	Prediction
a.	Predict the outcome of the model


1 Introduction
The goal of the competition is to design a machine learning algorithm that can automatically
sort short texts into a pre-determined set of topics. The provided dataset contains posts from
Reddit of 20 subreddits. The train set contains 3,500 messages and 1,500 messages as test set. In
the first phase of the project, we did a some text cleaning such as removing the stopwords to
make our text as clean as and use it for classification. The for the first milestone, we used numpy
to implement a pure Naive Bayse model and then later use laplace smoothing to make it more
accurate. In the last phase, that we had more flexibility, we first tried to use different approaches
to improve the result of Naive base classifier and then we tried to use other classifier such as
Random Forest and SVM.


2 Data Cleaning phase
The first step for our project. For the cleaning we did the following steps.
- Word tokenization and removing any charactors except numbers and alphabetical charactors.
- Remove numbers Remove any charactor which its length is less than two.
- Remove stop words.
- Steaming

3. Phase 1 of the project - Naive Bayse
As it was mentioned, the first phase was to use numpy for implementation of naive bayse
classifier and use the laplace smoothing approach to improve the accuracy of the model. we did
this part in two steps. The first step to implement the regular Naive bayse approch we gave us
the accuracy of 0.33647. However, we implemented Naive basye with Laplace smoothing and
we could get the accuracy of 0.55988.


4. Phase 2 of project - Naive Bayse, SVM and Random forest
For the second phase we used different approaches. At first, we used approaches that could
improve our Naive base approach and then we used skleran to implement more advanced
models such as Gaussian Naive Bayes, SVM and Random forest.

4.1 Improvement on Naive Bayse with Log probabilities
As it mentioned above, we could improve naive base with laplas smoothing that was implemneted
in the first phase. For that we used the Log Probabilities. Since probabilities are often small
numbers, To calculate joint probabilities, we need to multiply probabilities together ( as what
we did in the first phase). However, multiplying one small number by another small number,
get us very small numbers and make it difficult with the precision of floating point values, such
as under-runs. To avoid this problem, we used the log probability space (basically taking the
logarithm of probabilities).
This works because to make a prediction in Naive Bayes we need to know which class has the
larger probability (rank) rather than what the specific probability was. This approch improved
the accuracy of our model to 0.55866.

4.2 Use Distributions Gaussian and Bernoulli distribution
To use Naive Bayes with categorical attributes, we could calculate a frequency for each observation.
To use Naive Bayes with real-valued attributes, we can summarize the density of the
attribute using a Gaussian distribution. Alternatively we can use another functional form that
better describes the distribution of the data, such as an exponential. Using the Distributions
Gaussian gave us the best result we got which was 0.58604, however the bernoli distribution
result was 0.5445 which was not really an improvement.

4.3 SVM and Random Forest
We also used skleran to implement SVM, however with SVM we also did not get a better value
than Gaussian Distributions.


5 Comparison of the results
The table in page three, shows the comparison of the results for the model we implemented. It
should be said that the accuracy calculated for Random forest and SVM is not based on the test
set and it is based on the validation set that we had split from train set. As it is clear on the table,
the best result we achived was through using Naive bayse with Gaussian distribution.

Table 1: Comparison of the accuracy of different models
Models Liberary used for implementtion Public Score Private Score
1 Naive basye numpy 0.34909 0.35766
2 NB with Laplace numpy 0.55988 0.55223
3 NB Log Prob. numpy 0.56666 0.55866
4 NB Gaussian numpy, sklearn 0.59011 0.58604
5 NB Bernoull numpy, sklearn 0.54221 -
6 SVM sklearn 0.54011 -
7 Random Forest sklearn 0.48011 -
3
