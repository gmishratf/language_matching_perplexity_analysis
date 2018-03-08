import sys
import glob
import operator
import math
import decimal
from string import maketrans
from nltk import KneserNeyProbDist
from nltk import FreqDist
from nltk.util import ngrams

class BaseModel:
    ''' This class handles all core components of the system
        
        Every method's individual functionality is mentioned within
        
        Attribute definitions are as defined below:
        ### discount_KN is a float type discount value calculated during model_KN(), used for factoring discounts
            during perplexity calculation for Kneser-Ney smoothing based language models
        ### N is the value for n-gram length chosen for the system, decided through the type of language model 
            chosen. The method employed to tune and reach these values of N is described in readme.txt
        ### models{} is a dictionary of a dictionary. models{} has file names as keys, with a list of all n-grams
            in that file as the value, while this list serves as keys in the nested dictionary with corresponding
            probability values in a key-value format, depending on the model chosen for language matching
        ### training_file_names = [] is a list that stores all training file names, stores keys needed to iterate 
            over dict models{} 
    '''
    discount_KN = float
    N = int
    models = {}
    training_file_names = []
    perplexity_values = {}

    def __init__(self, typevar):
        ''' Values for n for each n-gram model, tuned for accuracy. Parameter for choosing values
            described in readme.txt
        '''
        if typevar == '--kneser-ney':
            self.N = 3
        elif typevar == '--unsmoothed':
            self.N = 1
        elif typevar == '--laplace':
            self.N = 3
        else:
            print("INVALID OPTION")

    def repair(self, text):
        ''' Repair function for following tasks:
            - replace " " with "_" to make spaces a valid token
            - remove punctuation from text
            - insert sentence beginning(@) and ending($) markers (training, dev and test files have been tested
              to check if they contain said markers as special characters
            - insert spaces between characters to "tokenize" each character
            - remove end of line characters
            - remove space padding at end of each text 
            - return repaired text as a word list of tokens
        '''
        text = text.translate(maketrans(" ", "_"))
        text_split = text.split('\n')
        for iter in range(len(text_split)):
            text_split[iter] = '@_' + text_split[iter] + '_$'
        text_split.pop()
        text = '_'.join(text_split)
        return text

    def model_unsmoothed(self, contents, list_ngrams, fdist_n, dict_fdist_n):
        ''' function returns an unsmoothed probability distribution (n-gram model) based on parameter
            list passed:
            - contents : list containing repaired contents of file whose n-gram model is to be created
            - list_ngrams : list containing n-grams for given model
            - fdist_n : frequency distribution for each n-gram created for given language
            - dict_fdist_n: dictionary containing n-grams and their corresponding count as a (key,value) pair
            
            This function uses the Markov assumption, i.e., P(wn | w1[n"1]) app. = P(wn | wn-N+1[n-1])
        '''
        ret_dict = {}
        if self.N == 1:
            length = len(contents)
            for iter in list_ngrams:
                ret_dict[iter] = dict_fdist_n[iter]/float(length)
        else:
            fdist_nminus_1 = FreqDist(list(ngrams(contents, self.N-1)))
            dict_fdist_nminus_1 = dict(fdist_nminus_1.items())
            for iter in list_ngrams:
                ret_dict[iter] = dict_fdist_n[iter]/float(dict_fdist_nminus_1[iter[:self.N-1]])
        return ret_dict

    def model_laplace(self, contents, list_ngrams, fdist_n, dict_fdist_n):
        ''' Function returns an add-one probability distribution (n-gram model) based on parameter list passed:
            - contents : list containing repaired contents of file whose n-gram model is to be created
            - list_ngrams : list containing n-grams for given model
            - fdist_n : frequency distribution for each n-gram created for given language
            - dict_fdist_n: dictionary containing n-grams and their corresponding count as a (key,value) pair
            
            This function uses the Markov assumption, i.e., P(wn | w1[n"1]) app. = P(wn | wn-N+1[n-1])

            V is the vocabulary count, i.e., number of all unique unigrams possible in the given language text
        '''
        ret_dict = {}
        V = len(set(contents))
        if self.N == 1:
            length = len(contents)
            for iter in list_ngrams:
                ret_dict[iter] = (dict_fdist_n[iter] + 1)/float(length + V)
        else:
            fdist_nminus_1 = FreqDist(list(ngrams(contents, self.N-1)))
            dict_fdist_nminus_1 = dict(fdist_nminus_1)
            for iter in list_ngrams:
                ret_dict[iter] = (dict_fdist_n[iter] + 1)/float(dict_fdist_nminus_1[iter[:self.N-1]] + V)
        return ret_dict

    def model_KN(self, contents):
        ''' function returns an unsmoothed probability distribution (n-gram model) based on parameter list
            passed:
            - contents : list containing repaired contents of file whose n-gram model is to be created
            
            Uses the KneserNeyProbDist() function from NLTK to create a Kneser-Ney smoothing based 
            language model
        '''
        ret_dict = {}
        list_ngrams = list(ngrams(contents, self.N))
        fdist = FreqDist(list_ngrams)
        kn_prob_dist = KneserNeyProbDist(fdist)
        self.discount_KN = kn_prob_dist.discount()
        for iter in kn_prob_dist.samples():
            ret_dict[iter] = kn_prob_dist.prob(iter)
        return ret_dict

    def create(self, contents, typevar):
        ''' list_ngrams is calculated from file contents, depending on type of model chosen from command line
            parameter list passed:
            - contens : list containing repaired contents of file whose n-gram model is to be created
            - typevar : type of smoothing language model chosen through command line

            This method creates a list of n-grams for the language, computes it's frequency distribution, maps the
            frequency distribution to a dictionary and passes these values to corresponding methods depending 
            on typevar
        '''
        list_ngrams = list(ngrams(contents, self.N))
        fdist_n = FreqDist(list_ngrams)
        dict_fdist_n = dict(fdist_n.items())
        ret_dict = {}
        if typevar == '--unsmoothed':
            ret_dict = self.model_unsmoothed(contents, list_ngrams, fdist_n, dict_fdist_n)
        elif typevar == '--laplace':
            ret_dict = self.model_laplace(contents, list_ngrams, fdist_n, dict_fdist_n)
        elif typevar == '--kneser-ney':
            ret_dict = self.model_KN(contents)
        return ret_dict

    def perplexity_unsmoothed(self, list_ngrams, model, length):
        ''' This method calculates perplexity for a test file over given unsmoothed language model passed to it as 
            a parameter.
            - list_ngrams: list of n-grams for given test file
            - model: unsmoothed n-gram model being tested for given test file
            - length: number of n-grams in test file

            Decimal data type has been used to improve precision

            #######__________IMPORTANT____________###################
            The method passes continue if it does not encounter a n-gram from the test file in given training model
            because math.log(0) is undefined and would not make sense, so we are ignoring n-grams which are not
            encountered in the training language model

            This is done to give somewhat feasible results using unsmoothed language models. If this is not needed,
            perplexity may be calculated using standard probabilities instead of log probabilities, which would end up
            giving infinite perplexity for OOV tokens.
        '''
        per = 0
        for iter in list_ngrams:
            if iter in model.keys():
                per = per + math.log(model[iter])
            else:
                continue
        return math.pow((1/decimal.Decimal(per).exp()), 1.0/length)

    def perplexity_laplace(self, list_ngrams, model, length):
        ''' This method calculates perplexity for a test file over given unsmoothed language model passed to it as 
            a parameter.
            - list_ngrams: list of n-grams for given test file
            - model: unsmoothed n-gram model being tested for given test file
            - length: number of n-grams in test file

            Decimal data type has been used to improve precision
            
            When an un-encountered n-gram is seen in the test set, instead of continuing or adding a zero probability,
            assign value of count = 1 and calculate probability for given n-gram. Also, count of all encountered n-grams
            has been increased by 1.

            This method is add-one smoothing. Accuracy may be improved by implementing add-k smoothing (i.e. true Laplacian
            smoothing)
        '''
        per = 0
        for iter in list_ngrams:
            if iter in model.keys():
                per = per + math.log(model[iter])
            else:
                per = per + math.log(1/decimal.Decimal((length)))
        return math.pow((1/decimal.Decimal(per).exp()), (1/decimal.Decimal(length)))

    def perplexity_KN(self, list_ngrams, model, length):
        ''' This method calculates perplexity for a test file over given unsmoothed language model passed to it as 
            a parameter.
            - list_ngrams: list of n-grams for given test file
            - model: unsmoothed n-gram model being tested for given test file
            - length: number of n-grams in test file

            Decimal data type has been used to improve precision
            
            Discount_KN has been calculated in model_KN() using KneserNeyProbDist().discount() from NLTK. Default value
            is 0.75
        '''
        per = 0
        for iter in list_ngrams:
            if iter in model.keys():
                per = per + math.log(model[iter])
            else:
                per = per + math.log(self.discount_KN/float(length))
        return math.pow(1/decimal.Decimal(per).exp(), 1.0/length)

    def perplexity(self, typevar, list_ngrams, model, length):
        ''' Calculates and returns value of perplexity for a given test list_ngrams over chosen model, selecting type of 
            model on basis of typevar passed through command line
        '''
        if typevar == '--unsmoothed':
            return self.perplexity_unsmoothed(list_ngrams, model, length)
        elif typevar == '--laplace':
            return self.perplexity_laplace(list_ngrams, model, length)
        elif typevar == '--kneser-ney':
            return self.perplexity_KN(list_ngrams, model, length)

    def get_file_contents(self, file_name):
        ''' Opens file with name passed as parameter, repairs it and returns contents of repaired file as a list
        '''
        f = open(file_name, 'r')
        return self.repair(f.read())

    def train(self, typevar):
        ''' Main training function
            Iterates over all files in source training folder and creates models for each file which is stored to models{}
        '''
        print("Training "+str(typevar)[2:]+" language models....")
        for iter in glob.glob('./811_a1_train/udhr-*.txt.tra'):
            print("."),
            self.training_file_names.append(iter)
            file_contents = self.get_file_contents(iter)
            self.models[iter] = self.create(file_contents, typevar)
    
    def test_dev(self, typevar):
        ''' Function to test all files in the dev set over language models created on basis of typevar passed through the
            command line
        '''
        f_unsmoothed = open('results_dev_unsmoothed.txt', 'w')
        f_laplace = open('results_dev_add-one.txt', 'w')
        f_KN = open('results_dev_kneser-ney.txt', 'w')
        print("\nEvaluating dev files....")
        for iter in glob.glob('./811_a1_dev/udhr-*.txt.dev'):
            print("."),
            test_file_contents = self.get_file_contents(iter)
            length = len(test_file_contents)
            list_ngrams = list(ngrams(test_file_contents, self.N))
            for x in self.training_file_names:
                self.perplexity_values[x] = self.perplexity(typevar, list_ngrams, self.models[x], length)
            mini = min(self.perplexity_values.iteritems(), key=operator.itemgetter(1))[0]
            if typevar == '--unsmoothed':
                print>>f_unsmoothed, str(iter[13:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
            elif typevar == '--laplace':
                print>>f_laplace, str(iter[13:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
            elif typevar == '--kneser-ney':
                print>>f_KN, str(iter[13:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
        f_unsmoothed.close()
        f_laplace.close()
        f_KN.close()

    def test_final(self, typevar):
        ''' Function to test all files in the test set over language models created on basis of typevar passed through the
            command line
        '''
        f_unsmoothed = open('results_test_unsmoothed.txt', 'w')
        f_laplace = open('results_test_laplace.txt', 'w')
        f_KN = open('results_test_kneser-ney.txt', 'w')
        print("\nEvaluating final test files....")
        for iter in glob.glob('./811_a1_test_final/udhr-*.txt.tes'):
            print("."),
            test_file_contents = self.get_file_contents(iter)
            length = len(test_file_contents)
            list_ngrams = list(ngrams(test_file_contents, self.N))
            for x in self.training_file_names:
                self.perplexity_values[x] = self.perplexity(typevar, list_ngrams, self.models[x], length)
            mini = min(self.perplexity_values.iteritems(), key=operator.itemgetter(1))[0]
            if typevar == '--unsmoothed':
                print>>f_unsmoothed, str(iter[20:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
            elif typevar == '--laplace':
                print>>f_laplace, str(iter[20:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
            elif typevar == '--kneser-ney':
                print>>f_KN, str(iter[20:]) + "   " + str(mini[15:]) + "   " + str(self.perplexity_values[mini]) + "   " + str(self.N)
        f_unsmoothed.close()
        f_laplace.close()
        f_KN.close()

def main():
    typevar = sys.argv[1]
    basemodel = BaseModel(typevar)
    basemodel.train(typevar)

    # comment out the next line if testing the dev set over language models is NOT required
    basemodel.test_dev(typevar)
    
    # comment out the next line if testing the final test set over language models is NOT required
    basemodel.test_final(typevar)

main()


