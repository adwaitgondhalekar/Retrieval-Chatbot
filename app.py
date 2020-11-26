#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,jsonify,render_template,request
import pickle
import numpy as np
import json
import sys
import random
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,Dropout, LSTM
import re
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


# In[2]:


with open("intents.json",encoding='utf-8') as file:
  data = json.load(file)


# In[3]:


input_sentences_tup=[]
for intent in data['intents']:
  for input_sent in intent['input']:
    input_sent=input_sent.lower()
    input_sentences_tup.append((input_sent,intent['tag']))
    


# In[4]:


input_sentences=[]
input_sentences_tag=[]
for sent,tag in input_sentences_tup:
  input_sentences.append(sent)
  input_sentences_tag.append(tag)


# In[5]:


le=LabelEncoder()
enc_labels=le.fit_transform(input_sentences_tag)


# In[6]:


labels=[]
for x in range(0,len(input_sentences_tag)):
  labels.append((enc_labels[x],input_sentences_tag[x]))


# In[7]:


def clean_text(line):
  line=re.sub(r"thats","that is",line)
  line=re.sub(r"whats","what is",line)
  line=re.sub(r"wheres","where is",line)
  line=re.sub(r"\'ll"," will",line)
  line=re.sub(r"\'ve"," have",line)
  line=re.sub(r"\'d"," would",line)
  line=re.sub(r"\'re"," are",line)
  line=re.sub(r"wouldnt","would not",line)
  line=re.sub(r"couldnt","could not",line)
  line=re.sub(r"shouldnt","should not",line)
  line=re.sub(r"wont","will not",line)
  line=re.sub(r"cant","can not",line)
  line=re.sub(r"[^\w\s]","",line)
  line=re.sub(r"im","i am",line)
  
  return line


# In[8]:


labels=dict(set(labels))


# In[9]:


no_of_labels=len(labels)


# In[10]:


enc_labels=enc_labels.reshape(len(enc_labels),1)


# In[11]:


one_hot=OneHotEncoder(sparse=False,dtype=int)
one_hot_labels=one_hot.fit_transform(enc_labels)


# In[12]:


tokenizer=Tokenizer(oov_token=1)
tokenizer.fit_on_texts(input_sentences)
total_words = len(tokenizer.word_index)+1


input_sequences=[]
for sents in input_sentences:
  sent_token=tokenizer.texts_to_sequences([sents])[0]
  input_sequences.append(sent_token)


max_seq_len=max([len(x) for x in input_sequences])



# In[13]:


app = Flask(__name__)

#from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
'''


tf_config = some_custom_config
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()


set_session(sess)'''
model=load_model('Retrieval_Chatbot_Model.h5')





# In[14]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])#,methods=['GET','POST'])
def predict():
  #input_text=request.form['input']
  input_text=None
  if request.method == "POST":
    input_text=request.form['input']
  #input_text=request.args.get('input')
  #print(msg,file=sys.stderr)
  #input_text=str(input_text)
  print("Input given is",input_text)
  

  input_text=input_text.lower()
  sent_token=tokenizer.texts_to_sequences([input_text])[0]
    #print(sent_token)
  sent_token=pad_sequences([sent_token],maxlen=max_seq_len,padding='pre')
    #print(sent_token)
    #print(model.predict(sent_token))
   
  highest_prob=np.max(model.predict(sent_token))


    
  arg=np.argmax(model.predict(sent_token))

    #print("prob-",highest_prob)
    #print("tag-",labels[arg]
  if highest_prob>0.50:
      #global sess
      #global graph
      
    predicted_class=model.predict_classes(sent_token)[0]

    for intent in data['intents']:
      if intent['tag']==labels[predicted_class]:
        responses=[]
        responses=intent['response']
        break

    #return render_template('index.html', prediction_text=random.choice(responses))
    response=random.choice(responses)
    return jsonify({'result':response})

  else:
    did_not_understand=["Sorry I didn't get you!","I did'nt get that!","Try something else","I don't understand!","Sorry try again!","Sorry what did you say?"]
    #return render_template('index.html', prediction_text=random.choice(did_not_understand))
    response=random.choice(did_not_understand)
    return jsonify({'result':response})
    

if __name__ == "__main__":
  app.run(debug=False,threaded=False)







