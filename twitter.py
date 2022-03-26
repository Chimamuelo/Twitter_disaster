import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download('stopwords')

stop_word=stopwords.words('english')


#Creating Dataframe
path='/Users/alber/Desktop/ML/NLPC/train_data.csv'
path_test='/Users/alber/Desktop/ML/NLPC/test_data.csv'
train_df=pd.read_csv(path)
test_df=pd.read_csv(path_test)
print(len(train_df['keyword'].unique()))
print(len(train_df['location'].unique()))
print(train_df['keyword'].value_counts())
print(train_df['location'].value_counts())

#Lower case string in column x
train_df['text']=train_df['text'].str.lower()
test_df['text']=test_df['text'].str.lower()
#replace all http[s]?:// combination with ' '
train_df['text'] = train_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
#print(type(train_df))
test_df['text'] = test_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
#Dropping irrelevant columns
#print(train_df['target'].value_counts())
train_df.drop(['id','keyword','location'],axis=1, inplace = True)
#print(train_df.head())
test_df.drop(['keyword','location'],axis=1, inplace = True)
#Creating sub datafframe
train_label=train_df['target']

text_train=train_df['text']


#remove stopwords from original df
filter_text=train_df['text'].apply(lambda x:' '.join(word for word in x.split() if word not in stop_word))
test_filter=test_df['text'].apply(lambda x:' '.join(word for word in x.split() if word not in stop_word))
training_labels=np.array(train_label)


X=round(.8*len(filter_text))




tokenizer=Tokenizer(15000,oov_token='<OOV>')
tokenizer.fit_on_texts(filter_text)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(filter_text)

x_train=pad_sequences(sequences,maxlen=50,padding='post',truncating='post')


print(len(sequences))

total_word=len(word_index)+1
print(total_word)

train_data=x_train[:X]
validation_data=x_train[X:]
y_train=training_labels[:X]
validation_label=training_labels[X:]




#testing



sequences_test=tokenizer.texts_to_sequences(test_filter)
x_test=pad_sequences(sequences_test,maxlen=50,padding='post',truncating='post')




model= tf.keras.Sequential([
    tf.keras.layers.Embedding(15000,32, input_length=50),
    #tf.keras.layers.Embedding(num_words, 128, input_length=max_content_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=False)),
    
    tf.keras.layers.Dense(32,activation='relu'),   
   
    tf.keras.layers.Dense(1,activation='sigmoid'),

    
   
    
    
    
   
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=.001),
loss='binary_crossentropy',
metrics=['accuracy'])




model.fit(train_data,y_train,epochs=100,validation_data=(validation_data,validation_label),batch_size=32,verbose=2,shuffle=True,
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)]

)

prediction=model.predict(x_test)


out=prediction.round().astype(int)
print(out)
final=test_df['id']
final=pd.concat([final,pd.DataFrame(data=out,columns=['target'])],axis=1)

print(final)

final.to_csv('Results6.csv',index=False)
















#print(text_train.head())
#print(type(text_train))






#sentence=train_df.loc[:5000, "text"].apply(lambda x: sent_tokenize(x))


#text_tokens=text_train.apply(word_tokenize)


#tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]




