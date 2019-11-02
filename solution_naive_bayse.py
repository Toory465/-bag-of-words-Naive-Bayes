import re
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# Classification Models
# train_data_path = '../data_train.pkl'
# test_data_path = '../data_test.pkl'
# final_output_path_nb = '../result_naive_bayse.csv'
# final_output_path_nb_lp = '../result_naive_bayse_lp_2.csv'

class Classifier:
    """
    The Classifier super class
    """
    def __init__(self):
        df_train, df_test = self._data_loader()
        self.train_data = df_train
        self.test_data = df_test
        self.number_of_classes = len(self.train_data['Id'].unique())
        self.topic_set = self.train_data['Id'].unique()

    def _data_loader(self):
        test_data = np.load(test_data_path , allow_pickle=True)
        train_data = np.load(train_data_path, allow_pickle=True)
        dt_train = pd.DataFrame(list(zip(train_data[0], train_data[1])), columns=['Category', 'Id'])
        dt_test = pd.DataFrame(test_data, columns=['Category'])
        return dt_train, dt_test

    def _csv_output_generator(self, prediction_vector, final_output_path):
        range_list = [item for item in range(0, len(self.test_data))]
        final_dt = pd.DataFrame(list(zip(prediction_vector, range_list)), columns=['Category', 'Id'])
        final_dt.to_csv(final_output_path)

class RandomClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def classifier(self):
        result_dt = self.test_data
        result_dt['Category'] = np.random.randint(0, self.number_of_classes, size=self.test_data.shape[0])
        return result_dt

class NaiveBayesClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.alpha = 2.15
        sent_set = set()
        for sentence in self.train_data['Category']:
            sent = re.sub("[^\w]", " ", sentence).split()
            list_ = [word for word in sent if word.isalnum()]
            for item in list_:
                sent_set.add(item)
        self.train_data_vocab_count = len(sent_set)

    def _word_extraction(self, sentence):
        stop_wrods = set(stopwords.words('english'))
        porter = PorterStemmer()
        words = re.sub("[^\w]", " ", sentence).split()
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if len(word) > 1]
        cleaned_text = [w.lower() for w in words if w.lower() not in stop_wrods]
        stemmed = [porter.stem(word) for word in cleaned_text]
        return stemmed

    def Naive_Bayse_probability_calculator(self):
        """
        Naive base classifer function
        :return:
        """
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

    def Naive_Bayse_probability_calculator_laplas_smothing(self):
        """
        naive base classifier with laplas smoothing function
        :return:
        """
        dict = {}
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

   
    def prediction(self, dict_):
        dt_test_category = self.test_data['Category'].apply(self._word_extraction)
        prediction_vector = []
        for i, sentence in enumerate(dt_test_category):
            score_dict = {}
            for topic in self.topic_set:
                topic_score = 1.0
                for word in sentence:
                    if dict_[topic][word] > 0.0 : 
                        probability = dict_[topic][word]
                    else:
                        probability = (1.0 / self.train_data_vocab_count)
                    topic_score = topic_score * probability
                    if topic_score == 0:
                        topic_score = 0.00001
                score_dict[topic] = topic_score
            prediction_vector.append(max(score_dict, key=score_dict.get))
        return prediction_vector

    def classifier(self):
        dt_train_category = self.train_data['Category'].apply(self._word_extraction)
        self.train_data['extract_words'] = dt_train_category

        # Naive bayse
        # dict_nb = self.Naive_Bayse_probability_calculator()
        # prediction = self.prediction(dict_nb)
        # self._csv_output_generator(prediction, final_output_path_nb)

        #Naive bayse Laplas classifier
        dict_nb_lp = self.Naive_Bayse_probability_calculator_laplas_smothing()
        prediction = self.prediction(dict_nb_lp)
        self._csv_output_generator(prediction, final_output_path_nb_lp) 


# In[104]:


nbc = NaiveBayesClassifier()
pred = nbc.classifier()

