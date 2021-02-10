#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tflearn as tl
import tensorflow as tf
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer


# In[2]:


with open("C:/Users/lokesh/Desktop/intents.json") as file:
    data = json.load(file)
    
print(data)


# In[3]:


nltk.download('punkt')


# In[4]:


stemmer = LancasterStemmer()
try:
    x
    with open("C:/Users/lokesh/Desktop/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    
except:    
    words = []
    tag = []
    doc_x = []
    doc_y = []
    training = []
    output = []

    for i in data["intents"]:
        for j in i["patterns"]:
            wrds = nltk.word_tokenize(j)
            words.extend(wrds)
            doc_x.append(wrds)
            doc_y.append(i["tag"])

            if i["tag"] not in tag:
                tag.append(i["tag"])

    
    words =  [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(tag)
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(doc_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(doc_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    training = np.array(training)
    output = np.array(output)
    
    with open("C:/Users/lokesh/Desktop/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)


# In[5]:


net = tl.input_data(shape= [None, len(training[0])])
net = tl.fully_connected(net, 8)
net = tl.fully_connected(net, 8)
net = tl.fully_connected(net, len(output[0]), activation = "softmax")
net = tl.regression(net)
model = tl.DNN(net)


# In[6]:


model.fit(training, output, n_epoch = 700, batch_size = 8, show_metric = True)


# In[7]:


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


# In[8]:


users = pd.read_csv("C:/Users/lokesh/Desktop/Ord.csv")
Name = input("Enter your Name\t")
k = ""
for i in range(len(users['Username'])):
    if Name == users['Username'][i]:
        k = Name
latestorder1 = users[(users['Username'] == k)]
latestorder = list(latestorder1[latestorder1['Date1'] == latestorder1['Date1'].max()]['Orders'])
latestorder    


# In[9]:


def chat():
    print("Bot: Hello "+Name+".. Welcome to Robo-Bot")
    while True:
        inp = input(Name+": \t")
        if inp.lower() == "quit":
            print("Bot: Thank you and Visit again")
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[0][results_index] >0.7:
            for tg in data["intents"]:
                if ((tag == "Menu") & (tg["tag"] == tag)):
                    responses = tg['responses']
                    print("Bot: What type of food you would like to order:")
                    for i in range(len(responses)):
                        print("Bot: Category:"+ str(i), responses[i] )
                       
                    inp_ord = input(Name+": \t")
                   
                    results_ord = model.predict([bag_of_words(inp_ord, words)])
                    results_index_ord = np.argmax(results_ord)
                    tag_ord = labels[results_index_ord]
                    
                    if results_ord[0][results_index_ord] >0.7:
                        for tg_ord in data["intents"]:
                            if (tg_ord["tag"] == tag_ord):
                                respt = tg_ord['responses']
                                print("Bot: Items under"+str(inp_ord)+" :\n", respt)
                                print("Bot: What would you like to eat?")
                               
                                int_ord = input(Name+": ")
                                resptl =[x.lower() for x in respt]
                                if int_ord.lower() in resptl:
                                    print("Bot: Would you like to order "+int_ord+". Press Yes to Confirm else No to Cancle")
                                    Confirm = input(Name+": ")
                                    if ((Confirm.lower() == "yes") | (Confirm.lower() == "yup") | (Confirm.lower() == "y") | (Confirm.lower() == "ok")  | (Confirm.lower() == "okay")  | (Confirm.lower() == "yeah")  | (Confirm.lower() == "sure")):
                                        print("Bot: Order Placed Successfully")
                                    else: print('Bot: Oder canclled')
                               
                elif ((tag == "Menu") or (tag == "ChiniesCourse") or (tag == "MainCourse") or (tag == "StarterList") or (tag == "SoupList")):
                    response2 = "Bot: Sorry!... I didn't get it, please try again"
                elif ((tag != "Menu") &(tag != "ChiniesCourse") & (tag != "MainCourse") & (tag != "StarterList") & (tag != "SoupList") & (tg["tag"] == tag)):
                    responses2 = tg['responses']
      
            if len(responses2) != 0:
                print("Bot: ",random.choice(responses2))
                responses2 = "\t"
        else:
            print("Bot: Sorry!... I didn't get it, please try again")      


# In[15]:


def chat_recomendation():
    print("Bot: Hello "+Name+"...! Welcome to Robo-Bot")
    while True:
        inp = input(Name+": \t")
        if inp.lower() == "quit":
            print("Bot: Thank you and Visit again")
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[0][results_index] >0.7:
            for tg in data["intents"]:
                if ((tag == "Menu") & (tg["tag"] == tag)):
                    
                    
                    responses = tg['responses']
                    print("Bot: Recommended food items based on your latest order, What would you like to eat?")
                    results1 = model.predict([bag_of_words(str(latestorder), words)])
                    results_index1 = np.argmax(results1)
                    tag1 = labels[results_index1]                  
                                                 
                    for tg_ord1 in data["intents"]:
                        if (tg_ord1["tag"] == tag1):
                            responses11 = tg_ord1['responses']
                    print("Bot: ",responses11)
                        
                    int_ord = input(Name+": ")
                    resptl =[x.lower() for x in responses11]
                    if int_ord.lower() in resptl:
                        print("Bot: Would you like to order "+int_ord+". Press Yes to Confirm else No to Cancle")
                        Confirm = input(Name+": ")
                        if ((Confirm.lower() == "yes") | (Confirm.lower() == "yup") | (Confirm.lower() == "y") | (Confirm.lower() == "ok")  | (Confirm.lower() == "okay")  | (Confirm.lower() == "yeah")  | (Confirm.lower() == "sure")):
                            print("Bot: Order Placed Successfully")
                        else: responses2 = 'Oder canclled'
                    elif int_ord.lower() == 'no':
                        if (tg["tag"] == tag):
                            responses_main = tg['responses']
                            print("Bot: What type of food you would like to order:")
                            for i in range(len(responses_main)):
                                print("Category:"+ str(i), responses_main[i] )
                                
                                
                                
                            inp_ord = input(Name+": \t")

                            results_ord = model.predict([bag_of_words(inp_ord, words)])
                            results_index_ord = np.argmax(results_ord)
                            tag_ord = labels[results_index_ord]

                            if results_ord[0][results_index_ord] >0.7:
                                for tg_ord in data["intents"]:
                                    if (tg_ord["tag"] == tag_ord):
                                        respt = tg_ord['responses']
                                        print("Bot: Items under"+str(inp_ord)+" :\n", respt)
                                        print("Bot: What would you like to eat?")

                                        int_ord = input(Name+": \t")
                                        resptl =[x.lower() for x in respt]
                                        if int_ord.lower() in resptl:
                                            print("Bot: Would you like to order "+int_ord+". Press Yes to Confirm else No to Cancle")
                                            Confirm = input(Name+": \t" )
                                            if ((Confirm.lower() == "yes") | (Confirm.lower() == "yup") | (Confirm.lower() == "y") | (Confirm.lower() == "ok")  | (Confirm.lower() == "okay")  | (Confirm.lower() == "yeah")  | (Confirm.lower() == "sure")):
                                                print("Bot: Order Placed Successfully")
                                            else: print('Bot: Oder canclled')                                
                            
                    
                    

                elif ((tag == "Menu") or (tag == "ChiniesCourse") or (tag == "MainCourse") or (tag == "StarterList") or (tag == "SoupList")):
                    response2 = "Sorry!... I didn't get it, please try again"
                elif ((tag != "Menu") &(tag != "ChiniesCourse") & (tag != "MainCourse") & (tag != "StarterList") & (tag != "SoupList") & (tg["tag"] == tag)):
                    responses2 = tg['responses']
      

            try: 
                print(random.choice(responses2))
                responses2 = "\t"
            except: print("Bot: Cyu: Order Cancelled and Type Quit to stop")

        else:
            print("Bot: Sorry!... I didn't get it, please try again")                                 


# In[16]:


if Name in list(users.Username):
    chat_recomendation()
else:
    chat()


# In[ ]:




