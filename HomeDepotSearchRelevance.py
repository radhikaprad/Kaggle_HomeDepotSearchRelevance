import os
import time
import nltk
import math
import collections
import csv
from math import*
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
import numpy as np

DESCRIPTIONS_FILE_PATH = 'C:\\homedepot\\product_descriptions.csv'
ATTRIBUTES_FILE_PATH = 'C:\\homedepot\\attributes.csv'
TRAININGDATA_FILE_PATH = 'C:\\homedepot\\train.csv'
TESTDATA_FILE_PATH = 'C:\\homedepot\\test.csv'
RESULT_FILE_PATH = 'C:\\homedepot\\result.csv'
ITERATION_COUNT_DEFAULT = 10000
ITERATION_COUNT_LIMIT = 1000000

filedescriptioncvs = open(DESCRIPTIONS_FILE_PATH, encoding="ISO-8859-1")
fileattributecvs = open(ATTRIBUTES_FILE_PATH, encoding="ISO-8859-1")
filetraincvs = open(TRAININGDATA_FILE_PATH, encoding="ISO-8859-1")

Pstemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
product_dictionary = {}

stop_words = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

# TermInfo namedtuple to store term info per term per file
TermInfo = collections.namedtuple("TermInfo", ["tf", "normalizedtfvalue", "tfidf", "normalizedtfidf"])

TRAINFILE_ID_COLUMNINDEX = 0
TRAINFILE_PRODUCTUID_COLUMNINDEX = 1
TRAINFILE_PRODUCTTITLE_COLUMNINDEX = 2
TRAINFILE_SEARCHTERM_COLUMNINDEX = 3
TRAINFILE_RELEVANCE_COLUMNINDEX = 4
DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX = 0
DESCRIPTIONFILE_DESCRIPTION_COLUMNINDEX = 1
ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX = 0
ATTRIBUTESFILE_NAME_COLUMNINDEX = 1
ATTRIBUTESFILE_VALUE_COLUMNINDEX = 2

attrcategories_index = {'title':1, 'brand':2,'description':3,'color':4,'bullets':5,'others':6,'material':7,'dimension':8,'weight':9} # Add new categories at end. Do not use '0' value index
attrcategories_count = len(attrcategories_index)
attrmapping_dict = { 'mfg brand name':attrcategories_index['brand'],'brand/model compatibility':attrcategories_index['brand'],'brand compatibility':attrcategories_index['brand'],'fits faucet brand':attrcategories_index['brand'],
                     'fits brands':attrcategories_index['brand'],'fits brand/models':attrcategories_index['brand'],'fits brands/models':attrcategories_index['brand'],'pump brand':attrcategories_index['brand'],
                     'fits brand/model':attrcategories_index['brand'],'brand/model/year compatibility':attrcategories_index['brand'],
                     
                     'color':attrcategories_index['color'],'color/finish':attrcategories_index['color'],'color family':attrcategories_index['color'],'color/finish family':attrcategories_index['color'],'fixture color/finish':attrcategories_index['color'],'fixture color/finish family':attrcategories_index['color'],'shade color family':attrcategories_index['color'],'actual color temperature (k)':attrcategories_index['color'],'color rendering index':attrcategories_index['color'],'top color family':attrcategories_index['color'],
                     
                     'bullet01':attrcategories_index['bullets'],'bullet02':attrcategories_index['bullets'],'bullet03':attrcategories_index['bullets'],'bullet04':attrcategories_index['bullets'],'bullet05':attrcategories_index['bullets'],'bullet06':attrcategories_index['bullets'],'bullet07':attrcategories_index['bullets'],'bullet08':attrcategories_index['bullets'],'bullet09':attrcategories_index['bullets'],'bullet10':attrcategories_index['bullets'],
                     
                     'material':attrcategories_index['material'],
                     
                     'product height (in.)':attrcategories_index['dimension'], 'product depth (in.)':attrcategories_index['dimension'], 
                     
                     'weight':attrcategories_index['weight']
                    }

def get_attributecategoryindex(attrname):
    attrname = attrname.lower()
    if attrname in attrmapping_dict:
        return (attrmapping_dict[attrname])
    else:
        return (attrcategories_index['others'])

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("deckover","deck over")
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([Pstemmer.stem(z) for z in s.split(" ")])
        #print(s)
        return s
    else:
        return "null"

def tf_value(count):
   return (1 + math.log10(count))

"""Unused Now - Should Try This Later"""
def normalizedtf_value(count, maxcount):
   return (count/maxcount)

def square_rooted(x):
   return (sqrt(sum([a*a for a in x])))

def normalizedcosinesimilarity(x,y):
    return (sum(a*b for a,b in zip(x,y)))

def stemmed_string(termsstr):
    termsstr = str_stem(termsstr)
    #print("Product Description: " + termsstr)
    wordtoken = tokenizer.tokenize(termsstr)
    stopw = [w for w in wordtoken if not w in stop_words ]
    stemmedwords = []
    for w in stopw:
        stemmedwords.append(Pstemmer.stem(w))
    return stemmedwords

def get_categoryinfodictionary(categorystring):
    stemmedwords = stemmed_string(categorystring)
    stemmedcounter = collections.Counter(stemmedwords)
    if len(stemmedcounter) == 0:
        return ""
    maxcountstr = stemmedcounter.most_common(1)[0][0]
    terminfodictionary = {}
    tflogarray = []
    for key in stemmedcounter.keys():
        tflog = tf_value(stemmedcounter[key])
        tflogarray.append(tflog)
        terminfodictionary[key] = TermInfo(stemmedcounter[key], normalizedtf_value(stemmedcounter[key],stemmedcounter[maxcountstr]), tflog, 0)
    normalizeddenominator = float(square_rooted(tflogarray))
    for key in terminfodictionary.keys():
        terminfo = terminfodictionary[key]
        terminfo = terminfo._replace(normalizedtfidf = (terminfo.tfidf/float(normalizeddenominator)))
        terminfodictionary[key] = terminfo
    return terminfodictionary

    """ DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX = 0
        DESCRIPTIONFILE_DESCRIPTION_COLUMNINDEX = 1
    """
def get_productdescription(uid):
    filedescriptioncvs.seek(0)
    filedescription_reader = csv.reader(filedescriptioncvs)
    next(filedescription_reader) # to skip first line that has the column name
    for row in filedescription_reader:
        if row[DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX] == uid: # description file has only one row per product id
            return row[DESCRIPTIONFILE_DESCRIPTION_COLUMNINDEX]
    return ""

    """ ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX = 0
        ATTRIBUTESFILE_NAME_COLUMNINDEX = 1
        ATTRIBUTESFILE_VALUE_COLUMNINDEX = 2
    """
def get_productattributes(uid, attribute_dictionary):
    fileattributecvs.seek(0)
    fileattributes_reader = csv.reader(fileattributecvs)
    next(fileattributes_reader) # to skip first line that has the column name
    attr_str = ''
    space_str = " "
    catdictionary = {}
    for row in fileattributes_reader:
        if row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX] is not '': # there are some empty rows
            if row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX] == uid:
                attrname = get_attributename(row[ATTRIBUTESFILE_NAME_COLUMNINDEX])
                attrcatindex = get_attributecategoryindex(attrname)
                attr_str = row[ATTRIBUTESFILE_VALUE_COLUMNINDEX]
                attr_str = attr_str.lower()
                if attr_str is 'yes':
                    attr_str = row[ATTRIBUTESFILE_NAME_COLUMNINDEX]
                elif attr_str is not 'no':
                    if attrcatindex in catdictionary:
                        catdictionary[attrcatindex] += space_str 
                        catdictionary[attrcatindex] += attr_str
                    else:
                        catdictionary[attrcatindex] = attr_str
                if int(row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX]) > int(uid): # the attributes are on sorted order on product ID. Optimize this later using binary search
                    break

    for attrcatindex in catdictionary:
        catstr = get_categoryinfodictionary(catdictionary[attrcatindex])
        if catstr is not "":
            attribute_dictionary[attrcatindex] = catstr

def build_product_dictionary():
    global product_dictionary
    print ('Started building product dictionary')
    filedescriptioncvs.seek(0)
    filedescription_reader = csv.reader(filedescriptioncvs)
    next(filedescription_reader) # to skip first line that has the column name
    fileattributecvs.seek(0)
    fileattributes_reader = csv.reader(fileattributecvs)
    next(fileattributes_reader) # to skip first line that has the column name
    attr_row = next(fileattributes_reader)  # Get first row
    space_str = " "
    for desc_row in filedescription_reader:
        if not attr_row:
            return # Looks like we have reached end of file
        descr_product_uid = desc_row[DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX]
        if descr_product_uid not in product_dictionary:
            attribute_dictionary = {}
            attribute_dictionary[attrcategories_index['description']] = get_categoryinfodictionary(desc_row[DESCRIPTIONFILE_DESCRIPTION_COLUMNINDEX])
            attr_product_uid = attr_row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX]

            while attr_product_uid is '': # there are some empty rows
                attr_row = next(fileattributes_reader)  # Move to next row
                if not attr_row:
                    return # Looks like we have reached end of file
                attr_product_uid = attr_row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX]
            catdictionary = {}
            while descr_product_uid == attr_product_uid:
                attrname = get_attributename(attr_row[ATTRIBUTESFILE_NAME_COLUMNINDEX])
                attrcatindex = get_attributecategoryindex(attrname)
                attr_str = attr_row[ATTRIBUTESFILE_VALUE_COLUMNINDEX]
                attr_str = attr_str.lower()
                if attr_str is 'yes':
                    attr_str = row[ATTRIBUTESFILE_NAME_COLUMNINDEX]
                elif attr_str is not 'no':
                    if attrcatindex in catdictionary:
                        catdictionary[attrcatindex] += space_str 
                        catdictionary[attrcatindex] += attr_str
                    else:
                        catdictionary[attrcatindex] = attr_str
                attr_row = next(fileattributes_reader)  # Move to next row
                if not attr_row:
                    break # Looks like we have reached end of file
                attr_product_uid = attr_row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX]
            for attrcatindex in catdictionary:
                catstr = get_categoryinfodictionary(catdictionary[attrcatindex])
                if catstr is not "":
                    attribute_dictionary[attrcatindex] = catstr
            product_dictionary[descr_product_uid] = attribute_dictionary
    print ('Completed building product dictionary')

def populate_dictionary(product_uid, product_title):
    global product_dictionary
    if product_uid not in product_dictionary:
        #print("populate_dictionary Getting file info for " + product_uid)
        attribute_dictionary = {}
        attribute_dictionary[attrcategories_index['title']] = get_categoryinfodictionary(product_title)
        attribute_dictionary[attrcategories_index['description']] = get_categoryinfodictionary(get_productdescription(product_uid))
        get_productattributes(product_uid,attribute_dictionary)
        product_dictionary[product_uid] = attribute_dictionary
    else:
        categoryinfo = product_dictionary[product_uid]
        if attrcategories_index['title'] not in categoryinfo:
            categoryinfo[attrcategories_index['title']] = get_categoryinfodictionary(product_title)

def get_productvector(queryterms,product_uid,product_title):
    product_uid = str(product_uid)
    populate_dictionary(product_uid, product_title)
    if product_uid not in product_dictionary:
        print ("Could not find product info in dictionary")
        return None

    querystemwords = stemmed_string(queryterms)
    stemmedcounter = collections.Counter(querystemwords)
    termcount = len(stemmedcounter)
    if termcount == 0:
        #print ('No query terms identified. Input query string:', queryterms)
        # There are some cases where pre-processing the search term does not give us anything  So, pick arbitrary (0.0001) as values for vector
        product_categoryvector = np.zeros(attrcategories_count + 1) # Plus 1 is for first column which will be set to 1 for 0th coefficient
        product_categoryvector.fill(0.0001)
        product_categoryvector[0] = 1 # Set first column to all 1s for 0th coefficient
        return None

    querytermsinfo = []
    for key in stemmedcounter.keys():
        tfvalue = tf_value(stemmedcounter[key])
        querytermsinfo.append(tfvalue)

    normalizedquerytermsinfo = []
    normalizeddenominator = float(square_rooted(querytermsinfo))
    for tfvalue in querytermsinfo:
        normalizedvalue = tfvalue / float(normalizeddenominator)
        normalizedquerytermsinfo.append(normalizedvalue)

    #print("NormalizedDenominator Query", normalizeddenominator)
    #print ("QueryTerms - Regular and Normalized")
    #print (querytermsinfo)
    #print (normalizedquerytermsinfo)
    #print ("FileTerms - Regular")
    #print (prodtermsinfo)

    product_categoryvector = np.zeros(attrcategories_count + 1) # Plus 1 is for first column which will be set to 1 for 0th coefficient
    product_categoryvector[0] = 1 # Set first column to all 1s for 0th coefficient
    categoryinfo = product_dictionary[product_uid]
    #print(categoryinfo)
    for attrcategory in attrcategories_index:
        categoryindex = attrcategories_index[attrcategory]
        similarityvalue = 0
        if categoryindex in categoryinfo:
            prodinfo = categoryinfo[categoryindex]
            if prodinfo is not None:
                prodtermsinfo = []
                for term in stemmedcounter.keys():
                    tfvalue = 0;
                    if term in prodinfo:
                        tfvalue = prodinfo[term].normalizedtfidf
                    prodtermsinfo.append(tfvalue)
                similarityvalue = normalizedcosinesimilarity(normalizedquerytermsinfo, prodtermsinfo)
        product_categoryvector[categoryindex] = similarityvalue # TODO Add weight to categories
    return product_categoryvector

class HomeDepotDataMining():
    def __init__(self, x_train=None, y_train=None):
        self.x_train = x_train
        self.y_train = y_train
        self.file_rowcount = 0
        self.alphas = [0, .1, .01, .001]
        self.preprocess_files() # Preprocess the files and build X and Y matrices once for training. This may take time

    def append_alpha(self, alpha_value): # For testing with different intial alpha values
        self.alphas.append(alpha_value)
        print (self.alphas)

    def reset_alphas(self):
        self.alphas = [0, .1, .01, .001]
        print (self.alphas)

    def clear_alphas(self):
        self.alphas = []
        print (self.alphas)

    """ TRAINFILE_ID_COLUMNINDEX = 0
        TRAINFILE_PRODUCTUID_COLUMNINDEX = 1
        TRAINFILE_PRODUCTTITLE_COLUMNINDEX = 2
        TRAINFILE_SEARCHTERM_COLUMNINDEX = 3
        TRAINFILE_RELEVANCE_COLUMNINDEX = 4
    """
    def preprocess_files(self):
        print ('Preprocessing product and training data files...')
        start_time = time.time()
        # Build initial product dictionary 
        build_product_dictionary()
        end_time = time.time()
        print('Product dictionary build time (secs): ', end_time- start_time)
        self.file_rowcount = get_train_rowcount() # Get row count from file
        # Build the X matrix from all rows of the training document
        start_time = time.time()
        filetraincvs.seek(0)
        traincsv_reader = csv.reader(filetraincvs)
        next(traincsv_reader) # to skip first line that has the column name
        index = 0;
        rowcount = self.file_rowcount
        self.x_train = np.zeros((rowcount, attrcategories_count + 1))
        self.y_train = np.zeros(rowcount)
        print ('Vectorizing Training DataSet...')
        for row in traincsv_reader:
            #print(row)
            prodvector = get_productvector(row[TRAINFILE_SEARCHTERM_COLUMNINDEX], row[TRAINFILE_PRODUCTUID_COLUMNINDEX], row[TRAINFILE_PRODUCTTITLE_COLUMNINDEX])
            if prodvector is not None:
                self.x_train[index] = prodvector
                self.y_train[index] = float(row[TRAINFILE_RELEVANCE_COLUMNINDEX])
                index += 1
            if index == rowcount:
                break
        end_time = time.time()
        # Close the files
        filedescriptioncvs.close()
        fileattributecvs.close()
        filetraincvs.close()
        print ('Training DataSet Size X', self.x_train.shape)
        print ('Training DataSet Size Y', self.y_train.shape)        
        print('Vectorizing Training DataSet time (secs): ', end_time- start_time)
        print ('Preprocessing complete. Ready for training.')

    def start_training(self, rowcount, num_iterations):
        start_time = time.time()
        if num_iterations == 0:
            num_iterations = ITERATION_COUNT_DEFAULT # Use default value if not specified
        min(num_iterations, ITERATION_COUNT_LIMIT) # Put a limit on iteration count to not keep looping forever
        rowcount = min(rowcount, self.file_rowcount)
        print ('Started Training...')
        if rowcount == 0 or rowcount == self.file_rowcount: # Train full file
            linear_regression = LinearRegression(x_train=self.x_train, y_train=self.y_train, initialtheta=1.0)
        else:
            iter_x_train = np.vsplit(self.x_train,[rowcount])[0]
            iter_y_train = np.split(self.y_train,[rowcount])[0]
            linear_regression = LinearRegression(x_train=iter_x_train, y_train=iter_y_train, initialtheta=1.0)
        # Train the model
        linear_regression.train(self.alphas, num_iterations) # Number of iterations
        end_time = time.time()
        print('Training time (secs): ', end_time- start_time)
        #linear_regression.update_initialtheta(0.5)
        #linear_regression.train(self.alphas, num_iterations) # Number of iterations
        start_time = time.time()
        linear_regression.generate_testpredictionsfile()
        end_time = time.time()
        print('Test file prediction time (secs): ', end_time- start_time)
        print ('Completed Training...')

def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class LinearRegression():
    def __init__(self, x_train=None, y_train=None, initialtheta = 0.5):
        self.x_train = x_train
        self.y_train = y_train
        self.m = y_train.shape[0]
        self.theta = np.empty(self.x_train.shape[1])
        self.theta.fill(initialtheta)

    def update_initialtheta(self, initialtheta): # For testing with different intial theta values
        self.theta.fill(initialtheta)

    def get_predictedvalue(self, x_vector):
        # Need to be in range [1,3]
        predicted_y = max(1.0,np.dot(x_vector, self.theta))
        predicted_y = min(3.0,predicted_y)
        return (predicted_y)

    def get_testrmse(self, x_validate, y_validate, theta):
        testcount = y_validate.shape[0]
        predicted_y = np.zeros(testcount)
        for i in range(0, testcount):
            predicted_y[i] = max(1.0,np.dot(x_validate[i,:], theta))
            predicted_y[i] = min(3.0,predicted_y[i]) # Need to be in range [1,3]
            #print('RMSE Actual Value:', y_validate[i],' Predicted Value:', predicted_y[i])
        return (calculate_rmse(predicted_y, y_validate))

    def gradient_descent(self, x_train, y_train, theta, alpha, num_iterations):
        m = y_train.shape[0]
        x_transpose = x_train.transpose()
        for iter in range(0, num_iterations):
            loss = np.dot(x_train, theta) - y_train
            J = np.sum(loss ** 2) / (2 * m)  # cost function
            #print ('Iter:', iter,'J:', J)
            gradient = np.dot(x_transpose, loss) / self.m         
            theta = theta - alpha * gradient  # update the theta value
        return theta

    def train(self, alphas, num_iterations):
        if len(alphas) == 0:
            print ('Alpha array is empty. Append alpha')

        print ('Started training with initial theta value:', self.theta[0])
        # Perform k-fold validation
        k_folds = 10
        best_rmse = 2.0 # starting with 2.0 error rate
        rowcount = self.x_train.shape[0]
        validate_size = rowcount / k_folds
        print ('Started k-fold cross validation')
        for i in range(0,k_folds):
            validate_startindex = i * validate_size
            validate_endindex = min(rowcount, validate_startindex + validate_size)
            # Split the input training matrix range:[0-k-l-n] to smaller sets
            if validate_startindex == 0: # [0-k Validation, k-n Training]
                split_x_arrays = np.vsplit(self.x_train,[validate_endindex])
                iter_x_validate = split_x_arrays[0]
                iter_x_train = split_x_arrays[1]
                split_y_arrays = np.split(self.y_train,[validate_endindex])
                iter_y_validate = split_y_arrays[0]
                iter_y_train = split_y_arrays[1]
            elif validate_endindex == rowcount: # [0-k Training, k-n Validation]
                split_x_arrays = np.vsplit(self.x_train,[validate_startindex])
                iter_x_validate = split_x_arrays[1]
                iter_x_train = split_x_arrays[0]
                split_y_arrays = np.split(self.y_train,[validate_startindex])
                iter_y_validate = split_y_arrays[1]
                iter_y_train = split_y_arrays[0]
            else: # [0-k Training, k-l Validation, l-n Training]
                split_x_arrays = np.vsplit(self.x_train,[validate_startindex,validate_endindex])
                iter_x_validate = split_x_arrays[1]
                iter_x_train = np.vstack((split_x_arrays[0],split_x_arrays[2]))
                split_y_arrays = np.split(self.y_train,[validate_startindex,validate_endindex])
                iter_y_validate = split_y_arrays[1]
                iter_y_train = np.concatenate((split_y_arrays[0],split_y_arrays[2]))

            print ('Perform k validation... Round:', (i + 1), ' X_train:', iter_x_train.shape, ' X_validate:', iter_x_validate.shape, ' Y_train:', iter_y_train.shape, ' Y_validate:', iter_y_validate.shape)
            print ('Best RMSE: ', best_rmse)
            for a in alphas:
                newtheta = self.gradient_descent(iter_x_train, iter_y_train, self.theta, a, num_iterations)
                rmse_value = self.get_testrmse(iter_x_validate, iter_y_validate, newtheta)
                if rmse_value < best_rmse:
                    best_rmse = rmse_value
                    self.theta = newtheta
                print ('Alpha: ', a,'Iterations:', num_iterations, 'RMSE:', rmse_value)
        # Run this the last time with whole training data
        print ('Perform training on whole data set...')
        best_rmse = 2.0  # Reset to 2.0 error rate for final run with whole data
        for a in alphas:
            newtheta = self.gradient_descent(self.x_train, self.y_train, self.theta, a, num_iterations)
            rmse_value = self.get_testrmse(self.x_train, self.y_train, newtheta)
            if rmse_value < best_rmse:
                best_rmse = rmse_value
                self.theta = newtheta
            print ('Alpha: ', a,'Iterations:', num_iterations, 'RMSE:', rmse_value)

    def generate_testpredictionsfile(self):
        with open(TESTDATA_FILE_PATH, encoding="ISO-8859-1") as filetestcvs:
            testcsv_reader = csv.reader(filetestcvs)
            next(testcsv_reader) # to skip first line that has the column name
            print ('Started generating test predictions file')
            testdata_X = np.zeros((1, attrcategories_count + 1)) # 1 row and attr categories column

            with open(RESULT_FILE_PATH, 'w', newline='') as filepredictions:
                filepredictionswriter = csv.writer(filepredictions, quoting=csv.QUOTE_ALL)
                filepredictionswriter.writerow(['id','relevance'])
                for row in testcsv_reader:
                    #print(row)
                    if row[TRAINFILE_PRODUCTUID_COLUMNINDEX] == '': # there are some empty rows
                        assert (0)
                    row_ID = row[TRAINFILE_ID_COLUMNINDEX]
                    prodvector = get_productvector(row[TRAINFILE_SEARCHTERM_COLUMNINDEX], row[TRAINFILE_PRODUCTUID_COLUMNINDEX], row[TRAINFILE_PRODUCTTITLE_COLUMNINDEX])
                    predictedvalue = 1 # Default to 1
                    if prodvector is not None:
                        predictedvalue = self.get_predictedvalue(prodvector)
                    filepredictionswriter.writerow([row_ID,predictedvalue])
        print ('Completed generating test predictions file result.csv')

""" CODE BELOW IS FOR ANALYSIS AND TESTING PURPOSE ONLY TO BE USED BT DEVELOPER"""

brand_attributes = ['mfg brand name', 'brand/model compatibility', 'brand compatibility', 'fits faucet brand', 'fits brands', 'fits brand/models', 'fits brands/models', 'pump brand', 'fits brand/model', 'brand/model/year compatibility']
color_attributes = ['color', 'color/finish', 'color family', 'color/finish family', 'fixture color/finish', 'fixture color/finish family', 'shade color family', 'actual color temperature (k)', 'color rendering index', 'top color family' ]
bullet_attributes = ['bullet01', 'bullet02', 'bullet03', 'bullet04', 'bullet05']
dimesions_attributes = ['product height (in.)', 'product depth (in.)']
weight_attributes = ['product weight (lb.)']

attrdict = {}
attrdict['brand'] = brand_attributes
attrdict['color'] = color_attributes
attrdict['bullet'] = bullet_attributes
attrdict['dimensions'] = dimesions_attributes
attrdict['weight'] = weight_attributes

def get_attributename(attrname):
    attrname = attrname.lower()
    for key in attrdict:
        if attrname in attrdict[key]:
            return (key)
    return (attrname)

    """ ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX = 0
        ATTRIBUTESFILE_NAME_COLUMNINDEX = 1
        ATTRIBUTESFILE_VALUE_COLUMNINDEX = 2
    """
def getuniqueattributes():
    fileattributecvs.seek(0)
    fileattributes_reader = csv.reader(fileattributecvs)
    next(fileattributes_reader) # to skip first line that has the column name
    uniqueattrs = {}
    for row in fileattributes_reader:
        attrname = get_attributename(row[ATTRIBUTESFILE_NAME_COLUMNINDEX])
        if attrname not in uniqueattrs:
            uniqueattrs[attrname] = 1
        else:
            uniqueattrs[attrname] += 1
    for key, value in sorted(uniqueattrs.items(), key=lambda item: (item[1], item[0]), reverse=True):
        print ("%s: %s" % (key, value))

def get_train_rowcount():
    with open(TRAININGDATA_FILE_PATH, encoding="ISO-8859-1") as filetrain_rowcount:
        traincsv_reader = csv.reader(filetrain_rowcount)
        next(traincsv_reader) # to skip first line that has the column name
        data = list(traincsv_reader)
        row_count = len(data)
        print ('Train Data Row Count:', row_count)
        return (row_count)

def get_test_rowcount():
    with open(TESTDATA_FILE_PATH, encoding="ISO-8859-1") as filetest_rowcount:
        testcsv_reader = csv.reader(filetest_rowcount)
        next(testcsv_reader) # to skip first line that has the column name
        data = list(testcsv_reader)
        row_count = len(data)
        print ('Test Data Row Count:', row_count)

def get_result_rowcount():
    if os.path.isfile(RESULT_FILE_PATH):
        with open(RESULT_FILE_PATH, encoding="ISO-8859-1") as fileresult_rowcount:
            resultcsv_reader = csv.reader(fileresult_rowcount)
            next(resultcsv_reader) # to skip first line that has the column name
            data = list(resultcsv_reader)
            row_count = len(data)
            print ('Result Data Row Count:', row_count)
    else:
        print ('Result file does not exist')

def test_files_sorted():
    fileattributecvs.seek(0)
    fileattributes_reader = csv.reader(fileattributecvs)
    next(fileattributes_reader) # to skip first line that has the column name
    lastrow_productid = 0
    for row in fileattributes_reader:
        if row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX] is not '': # there are some empty rows
            if int(row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX]) < lastrow_productid: # the attributes are on sorted order on product ID. Optimize this later using binary search
                print ('ATTRIBUTES FILE - NOT SORTED LastRow_ProductID', lastrow_productid,' CurrentRow_ProductID ', row[ATTRIBUTESFILE_PRODUCTUID_COLUMNINDEX])
                return
    print ('ATTRIBUTES FILE - SORTED !!!')
    filedescriptioncvs.seek(0)
    filedescription_reader = csv.reader(filedescriptioncvs)
    next(filedescription_reader) # to skip first line that has the column name
    lastrow_productid = 0
    for row in filedescription_reader:
        if row[DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX] is not '': # there are some empty rows
            if int(row[DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX]) < lastrow_productid:
                print ('DESCRIPTION FILE - NOT SORTED LastRow_ProductID', lastrow_productid,' CurrentRow_ProductID ', row[DESCRIPTIONFILE_PRODUCTUID_COLUMNINDEX])
                return
    print ('DESCRIPTION FILE - SORTED !!!')
