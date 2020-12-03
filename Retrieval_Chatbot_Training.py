
import numpy as np
import json
import random
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,Dropout, LSTM
import re


with open("intents.json",encoding='utf-8') as file:
  data = json.load(file)


input_sentences_tup=[]
for intent in data['intents']:
  for input_sent in intent['input']:
    input_sent=input_sent.lower()
    input_sentences_tup.append((input_sent,intent['tag']))
   

input_sentences_tup


input_sentences=[]
input_sentences_tag=[]
for sent,tag in input_sentences_tup:
  input_sentences.append(sent)
  input_sentences_tag.append(tag)



input_sentences



input_sentences_tag


le=LabelEncoder()
enc_labels=le.fit_transform(input_sentences_tag)



enc_labels



labels=[]
for x in range(0,len(input_sentences_tag)):
  labels.append((enc_labels[x],input_sentences_tag[x]))



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




labels



labels=dict(set(labels))



labels



no_of_labels=len(labels)


enc_labels=enc_labels.reshape(len(enc_labels),1)


enc_labels



one_hot=OneHotEncoder(sparse=False,dtype=int)
one_hot_labels=one_hot.fit_transform(enc_labels)


one_hot_labels


tokenizer=Tokenizer(oov_token=1)
tokenizer.fit_on_texts(input_sentences)
total_words = len(tokenizer.word_index)+1


total_words


input_sequences=[]
for sents in input_sentences:
  sent_token=tokenizer.texts_to_sequences([sents])[0]
  input_sequences.append(sent_token)



input_sequences



max_seq_len=max([len(x) for x in input_sequences])


max_seq_len



input_sequences=pad_sequences(input_sequences,max_seq_len,padding='pre')



input_sequences



input_sequences=np.array(input_sequences)


X=input_sequences
Y=one_hot_labels




X



Y



X=np.array(X)
Y=np.array(Y)




model=Sequential()
model.add(Embedding(total_words,64,input_length=max_seq_len))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128))
model.add(Dropout(rate=0.2))
model.add(Dense(no_of_labels,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




model.summary()




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




