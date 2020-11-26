#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import random
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,Dropout, LSTM
import re


# In[3]:


with open("intents.json",encoding='utf-8') as file:
  data = json.load(file)


# In[4]:


input_sentences_tup=[]
for intent in data['intents']:
  for input_sent in intent['input']:
    input_sent=input_sent.lower()
    input_sentences_tup.append((input_sent,intent['tag']))
    


# In[5]:


input_sentences_tup


# In[6]:


input_sentences=[]
input_sentences_tag=[]
for sent,tag in input_sentences_tup:
  input_sentences.append(sent)
  input_sentences_tag.append(tag)


# In[7]:


input_sentences


# In[8]:


input_sentences_tag


# In[9]:


le=LabelEncoder()
enc_labels=le.fit_transform(input_sentences_tag)


# In[10]:


enc_labels


# In[11]:


labels=[]
for x in range(0,len(input_sentences_tag)):
  labels.append((enc_labels[x],input_sentences_tag[x]))


# In[12]:


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


# In[13]:


labels


# In[14]:


labels=dict(set(labels))


# In[15]:


labels


# In[16]:


no_of_labels=len(labels)


# In[17]:


enc_labels=enc_labels.reshape(len(enc_labels),1)


# In[18]:


enc_labels


# In[19]:


one_hot=OneHotEncoder(sparse=False,dtype=int)
one_hot_labels=one_hot.fit_transform(enc_labels)


# In[20]:


one_hot_labels


# In[21]:


tokenizer=Tokenizer(oov_token=1)
tokenizer.fit_on_texts(input_sentences)
total_words = len(tokenizer.word_index)+1


# In[22]:


total_words


# In[23]:


input_sequences=[]
for sents in input_sentences:
  sent_token=tokenizer.texts_to_sequences([sents])[0]
  input_sequences.append(sent_token)


# In[24]:


input_sequences


# In[25]:


max_seq_len=max([len(x) for x in input_sequences])


# In[26]:


max_seq_len


# In[27]:


input_sequences=pad_sequences(input_sequences,max_seq_len,padding='pre')


# In[28]:


input_sequences


# In[29]:


input_sequences=np.array(input_sequences)


# In[30]:


X=input_sequences
Y=one_hot_labels


# In[31]:


X


# In[32]:


Y


# In[33]:


X=np.array(X)
Y=np.array(Y)


# In[34]:


model=Sequential()
model.add(Embedding(total_words,64,input_length=max_seq_len))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(64))
model.add(Dense(no_of_labels,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[35]:


model.summary()


# In[36]:


model.fit(X,Y,epochs=50,shuffle=True)
model.save('Retrieval_Chatbot_Model.h5')





# In[37]:


'''
def response_gen(input_text):
  input_text=input_text.lower()
  sent_token=tokenizer.texts_to_sequences([input_text])[0]
  #print(sent_token)
  sent_token=pad_sequences([sent_token],maxlen=max_seq_len,padding='pre')
  #print(sent_token)
  #print(model.predict(sent_token))
  highest_prob=np.max(model.predict(sent_token))
  arg=np.argmax(model.predict(sent_token))
  print("prob-",highest_prob)
  print("tag-",labels[arg])

  if highest_prob>0.50:
    predicted_class=model.predict_classes(sent_token)[0]

    for intent in data['intents']:
      if intent['tag']==labels[predicted_class]:
        responses=[]
        responses=intent['response']
        break
  
    return random.choice(responses),predicted_class

  else:
    did_not_understand=["Sorry I didn't get you!","I did'nt get that!","Try something else","I don't understand!","Sorry try again!","Sorry what did you say?"]
    return random.choice(did_not_understand),-1
    '''


# In[39]:


'''
print("Human Chat with me!")
while True:
  user_inp=input()
  cleaned_user_inp=clean_text(user_inp)
  #print(cleaned_user_inp)
  response,tag=response_gen(cleaned_user_inp)
  print(response)
  if tag==5:
    break
'''    
  


# In[ ]:




