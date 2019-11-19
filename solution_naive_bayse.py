import re
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import math

## Skleran liberary for the second phayse of the competition
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

# Classification Models
train_data_path = './data_train.pkl'
test_data_path = '../data_test.pkl'
final_output_path_nb = '../result_naive_bayse.csv'
final_output_path_nb_lp = '../result_naive_bayse_lp.csv'
final_output_path = '../result.csv'


### the  Parent Classifier class
class Classifier:
    def __init__(self):
        df_train, df_test = self._data_loader()
        self.train_data = df_train
        self.test_data = df_test
        self.number_of_classes = len(self.train_data['Id'].unique())
        self.topic_set = self.train_data['Id'].unique()

    def _data_loader(self):
        test_data = np.load(test_data_path, allow_pickle=True)
        train_data = np.load(train_data_path, allow_pickle=True)
        dt_train = pd.DataFrame(list(zip(train_data[0], train_data[1])), columns=['Category', 'Id'])
        dt_test = pd.DataFrame(test_data, columns=['Category'])
        return dt_train, dt_test

    def _csv_output_generator(self, prediction_vector, final_output_path):
        range_list = [item for item in range(0, len(self.test_data))]
        final_dt = pd.DataFrame(list(zip(prediction_vector, range_list)), columns=['Category', 'Id'])
        final_dt.to_csv(final_output_path)


# the Random Classifier
class RandomClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def classifier(self):
        result_dt = self.test_data
        result_dt['Category'] = np.random.randint(0, self.number_of_classes, size=self.test_data.shape[0])
        return result_dt


## The Naive base Classifier
class NaiveBayesClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.alpha = 2.25
        sent_set = set()
        for sentence in self.train_data['Category']:
            sent = re.sub("[^\w]", " ", sentence).split()
            list_ = [word for word in sent if word.isalnum()]
            for item in list_:
                sent_set.add(item)
        self.train_data_vocab_count = len(sent_set)
        self.Encoder = LabelEncoder()

    # Data Cleaning part
    def _word_extraction(self, sentence):
        stop_wrods = set(stopwords.words('english'))
        porter = PorterStemmer()
        words = re.sub("[^\w]", " ", sentence).split()
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if len(word) > 1]
        cleaned_text = [w.lower() for w in words if w.lower() not in stop_wrods]
        stemmed = [porter.stem(word) for word in cleaned_text]
        return stemmed

    # implementaiton of simple Naive Bayse CLassifier
    def Naive_Bayse_probability_calculator(self):
        dict = {}
        for topic in self.topic_set:
            dt = self.train_data[self.train_data['Id'] == topic]
            topic_words = [item for row in dt['extract_words'] for item in row]
            word_counts = len(topic_words)
            unique_words_count = len(set(topic_words))
            topic_freq_series = FreqDist(topic_words)
            for key, value in topic_freq_series.items():
                topic_freq_series[key] = value / word_counts
            dict[topic] = topic_freq_series
        return dict

    ## Implementation of Naive Bayse Classifier with Laplace smoothing
    def Naive_Bayse_probability_calculator_laplas_smothing(self):
        dict = {}
        topic_word_count = {}
        for topic in self.topic_set:
            dt = self.train_data[self.train_data['Id'] == topic]
            topic_words = [item for row in dt['extract_words'] for item in row]
            word_counts = len(topic_words)
            unique_words_count = len(set(topic_words))
            topic_freq_series = FreqDist(topic_words)
            for key, value in topic_freq_series.items():
                topic_freq_series[key] = ((value + self.alpha) / (word_counts + (self.alpha * unique_words_count)))
            dict[topic] = topic_freq_series
        return dict

    # The prediction Method
    def prediction(self, dict_):
        dt_test_category = self.test_data['Category'].apply(self._word_extraction)
        prediction_vector = []
        for i, sentence in enumerate(dt_test_category):
            score_dict = {}
            for topic in self.topic_set:
                topic_score = 0.0

                # topic_score = 1.0    /// NAive bayse approch
                for word in sentence:
                    if dict_[topic][word] > 0.0:
                        probability = dict_[topic][word]
                    else:
                        probability = (1.0 / self.train_data_vocab_count)

                    ### Caclulate Log probabilities
                    topic_score = topic_score + math.log(probability)

                    ### Caclulate pure naive bayse appraoch
                    ## topic_score = topic_score * probability
                #                     if topic_score == 0:  /// NAive bayse approch
                #                         topic_score = 0.00001  /// NAive bayse approch
                score_dict[topic] = topic_score
            prediction_vector.append(max(score_dict, key=score_dict.get))
        return prediction_vector

    # the classifier method
    def classifier(self):
        dt_train_category = self.train_data['Category'].apply(self._word_extraction)
        self.train_data['extract_words'] = dt_train_category

        # Naive bayse
        # dict_nb = self.Naive_Bayse_probability_calculator()
        # prediction = self.prediction(dict_nb)
        # self._csv_output_generator(prediction, final_output_path_nb)

        # Naive bayse Laplas classifier
        dict_nb_lp = self.Naive_Bayse_probability_calculator_laplas_smothing()
        prediction = self.prediction(dict_nb_lp)
        self._csv_output_generator(prediction, final_output_path_nb_lp)

    def _sklearn_data_cleaning(self):
        Train_Y = self.Encoder.fit_transform(self.train_data['Id'])
        Tfidf_vect = TfidfVectorizer()
        Tfidf_vect.fit(self.train_data['Category'])
        Train_X_Tfidf = Tfidf_vect.transform(self.train_data['Category'])
        Test_X_Tfidf = Tfidf_vect.transform(self.test_data['Category'])
        return Train_X_Tfidf, Test_X_Tfidf, Train_Y

    def guassian_distribution_classifier(self):

        Train_X_Tfidf, Test_X_Tfidf, Train_Y = self._sklearn_data_cleaning()
        gnb = ComplementNB(alpha=1.590)
        gnb.fit(Train_X_Tfidf, Train_Y)
        # predict the labels on validation dataset
        predictions_NB = gnb.predict(Test_X_Tfidf)

        range_list = [item for item in range(0, len(self.test_data))]
        final_dt = pd.DataFrame(list(zip(self.Encoder.inverse_transform(predictions_NB), range_list)),
                                columns=['Category', 'Id'])
        self._csv_output_generator(final_dt, final_output_path)

        ## Support Vecto Machine

    def SVM(self):

        Train_X_Tfidf, Test_X_Tfidf, Train_Y = self._sklearn_data_cleaning()
        model = LinearSVC()
        model.fit(Train_X_Tfidf, Train_Y)

        # predict the labels on validation dataset
        predictions_NB = model.predict(Test_X_Tfidf)

        range_list = [item for item in range(0, len(self.test_data))]
        final_dt = pd.DataFrame(list(zip(self.Encoder.inverse_transform(predictions_NB), range_list)),
                                columns=['Category', 'Id'])
        self._csv_output_generator(final_dt, final_output_path)


## define the classifier object
nbc = NaiveBayesClassifier()

### NAive bayse classifier using laplac smoothing and log probabilities
pred = nbc.classifier()

## Classification using Guassian Distribution
pred = nbc.guassian_distribution_classifier()

## CLasiification Using SVM
pred = nbc.SVM()