#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, flash , redirect, render_template , request, session, abort , Markup
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from werkzeug import secure_filename
import json
from scipy.stats import kurtosis, skew
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# In[2]:


app = Flask(__name__)
app.secret_key = os.urandom(12)

dropdown_list = []
dropdown_list_2 = []


# In[3]:


@app.route('/upload')
def upload_file():
    dropdown_list.clear()
    dropdown_list_2.clear()
    return render_template('upload.html')


# In[4]:


@app.route('/input_percent' , methods = ['GET','POST'])
def input_num():
    x = request.form["in"]
    print("dsjfkkdsjf",x)
    fpath = os.path.join("default", "testtestreason1.csv")
    line = pd.read_csv(fpath).shape[0]
    y = round((float(x)*line)/100)
    print(line)
    print(y)
    ls = []
    lschurn=[]
    with open(fpath) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= y and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv(fpath)['Exited'][:y].copy())))  
    print(lss)
    print(lschurn)
    return render_template('Percent.html', outList = ls, value_list=lschurn,values_res=lss )


# In[5]:


@app.route('/check/<string:dropdown>',methods=['POST','GET'])
def specific(dropdown):
    x = dropdown
    yo = predict_default(x)
    x = search_default(x)
    rownum  = x[0]
    ccid = x[1]
    surname = x[2]
    creditscore  = x[3]
    geo = x[4]
    gender  = x[5]
    age  = x[6]
    tenure = x[7]
    balance  = x[8]
    numpro = x[9]
    hascard = x[10]
    activemem = x[11]
    salary = x[12]    
    x = x[13]
    pred= float(x)*100
    labels = ["probability",""]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    colors = [ "#F7464A", "#46BFBD", "#FDB45C" , "#ABCDEF"]
    return render_template('Chart.html', set=zip(values_res, labels_res, colors),firstname=x, 
                           rownum=rownum, ccid=ccid, surname=surname, creditscore=creditscore, 
                           geo=geo, gender=gender, age=age, tenure=tenure, balance=balance, 
                           numpro=numpro, hascard = hascard, activemem = activemem, salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , secondname2 = values_res[2] , 
                           secondname3 = values_res[3] ,labels_res=labels_res,values_res=values_res, values=values, labels=labels)


# In[6]:


def preprocess_data(fileInput):
    dataset = pd.read_csv(fileInput)
    dataset['Geography'] = dataset['Geography'].astype('category')
    dataset['Gender'] = dataset['Gender'].astype('category')
    dataset['EstimatedSalary'] = dataset['EstimatedSalary'].astype('float32')
    dataset['Balance'] = dataset['Balance'].astype('float32')
    cat_columns = dataset.select_dtypes(['category']).columns
    dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
    X = dataset.iloc[:, 3:13].values
    Y = dataset.iloc[:, 13].values
    sc = StandardScaler()
    X_test = sc.fit_transform(X)
    return X_test
    
    


# In[7]:


@app.route('/')
def home():
        return render_template('upload.html')


# In[8]:


def uploadFileAndPredict(filename):
    K.clear_session()
    dropdown_list_2.clear()
    proceseed_data = preprocess_data(filename)
    predictChurnAndReasons(proceseed_data,filename)
    
    return proceseed_data          
    


# In[9]:


def predictChurnAndReasons(proceseed_data,filename):
    model = pickle.load(open('SVCclassifier.h5', 'rb'))
    y_pred = model.predict_proba(proceseed_data)
    df = pd.read_csv(filename)
    df['Exited'] = y_pred
    df.set_index('RowNumber', inplace=True)
    df.sort_values('Exited', ascending=False, inplace=True)
    fpathr = os.path.join("default", "testtestreason1.csv")
    df.to_csv(fpathr)  
    return y_pred
    
    


# In[10]:


@app.route('/defaultfile', methods = ['GET', 'POST'])
def uploader_default_file():
    fpath = os.path.join("default", "testtestdefault1.csv")
    uploadFileAndPredict(fpath)
    with open(fpath) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount==0:
                lineCount=lineCount+1
            else:
                lineCount=lineCount+1
                dropdown_list_2.append((row[1]))
    return render_template('Result.html',  dropdown_list_2=dropdown_list_2)


# In[11]:


@app.route('/check_default/<string:dropdown_2>',methods=['POST','GET'])
def specific_default(dropdown_2):
    x = dropdown_2
    yo = predict_default(x)
    x = search_default(x)
    rownum  = x[0]
    ccid = x[1]
    surname = x[2]
    creditscore  = x[3]
    geo = x[4]
    gender  = x[5]
    age  = x[6]
    tenure = x[7]
    balance  = x[8]
    numpro = x[9]
    hascard = x[10]
    activemem = x[11]
    salary = x[12]
    x = x[13]
    pred= float(x)*100
    labels = ["probability",""]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    colors = [ "#F7464A", "#46BFBD", "#FDB45C" , "#ABCDEF"]
    return render_template('Chart.html', set=zip(values_res, labels_res, colors),
                           firstname=x, rownum=rownum, ccid=ccid, surname=surname, 
                           creditscore=creditscore, geo=geo, gender=gender, age=age, 
                           tenure=tenure, balance=balance, numpro=numpro, hascard = hascard, activemem = activemem, 
                           salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , 
                           secondname2 = values_res[2] , secondname3 = values_res[3] ,labels_res=labels_res,
                           values_res=values_res, values=values, labels=labels)


# In[12]:


@app.route("/predict_default", methods=["GET","POST"])
def predict_default(z):
    K.clear_session()
     # output file
    cid1 = z 
    test3 = model_default_2(cid1)
    model2 = load_model('my_model2.h5')
    model2._make_predict_function()
    y_pred2 = model2.predict(test3)
    resons=["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    dic=dict()
    diff=list()
    for j in range(len(y_pred2)):
        dic.clear()
    for (label, p) in zip(resons, y_pred2[j]):
        dic[label]= p*100
    diff.append(dic.copy())
    j = json.dumps(diff)


    return y_pred2


# In[13]:


def model_default_2(cid1):
    
    fpathr = os.path.join("default", "testtestreason1.csv")
    test = pd.read_csv(fpathr)
    cid1 = int(cid1)

    X_test=test.loc[test['CustomerId']==cid1].values.copy()
    X_test=X_test[:, 3:14]
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_3 = LabelEncoder()
    X_test[:,1] = labelencoder_X_3.fit_transform(X_test[:, 1])
    labelencoder_X_4 = LabelEncoder()
    X_test[:,2] = labelencoder_X_4.fit_transform(X_test[:, 2])      
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)

    return X_test


# In[14]:


def search_default(cid):
    fpathr = os.path.join("default", "testtestreason1.csv")
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        for row in allRead:
            if row[1]==cid:
                return row


# In[ ]:


if __name__ == "__main__":
    app.run()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




