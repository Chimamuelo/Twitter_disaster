import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


nltk.download('stopwords')

stop_word=stopwords.words('english')


#Creating Dataframe
path='/Users/alber/Desktop/ML/NLPC/train_data.csv'
path_test='/Users/alber/Desktop/ML/NLPC/test_data.csv'
train_df=pd.read_csv(path)
test_df=pd.read_csv(path_test)
print(train_df.head(100))

#Lower case string in column x
train_df['text']=train_df['text'].str.lower()
test_df['text']=test_df['text'].str.lower()
#replace all http[s]?:// combination with ' '
train_df['text'] = train_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
#print(type(train_df))
test_df['text'] = test_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
#Dropping irrelevant columns

#print(train_df.head())

#Creating sub datafframe
train_label=train_df['target']






#remove stopwords from original df
filter_text=train_df['text'].apply(lambda x:' '.join(word for word in x.split() if word not in stop_word))
test_filter=test_df['text'].apply(lambda x:' '.join(word for word in x.split() if word not in stop_word))
training_labels=np.array(train_label)

train_df['text']=filter_text





X=round(.8*len(filter_text))
train_df.drop(['target'],axis=1,inplace=True)
train_df.drop(['id'],axis=1,inplace=True)

#train_data=filter_text[:X]

#validation_data=filter_text[X:]
#train_l=training_labels[:X]
#validation_label=training_labels[X:]


train_data=train_df[:X]
validation_data=train_df[X:]
train_l=training_labels[:X]
validation_l=training_labels[X:]



tokenizer=Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
word_index=tokenizer.word_index

lenghts=[len(t.split(' ')) for t in train_data]
plt.hist(lenghts,bins=len(set(lenghts)))
plt.show()

maxlen=50

def get_sequences(tokenizer,tweets):
  sequences= tokenizer.texts_to_sequences(tweets)
  padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=50)
  return padded

padded_train=get_sequences(tokenizer,train_data)
padded_val=get_sequences(tokenizer,validation_data)



#print(train_data.shape)


#train_data['text']=padded












model= tf.keras.Sequential([
 tf.keras.layers.Embedding(10000,16,input_length=maxlen),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
  tf.keras.layers.Dense(1,activation='sigmoid')

]
)




model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=.001),
loss='binary_crossentropy',
metrics=['accuracy'])

model.fit(padded_train,train_l,
epochs=100,
validation_data=(validation_data,validation_l),
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)]


)

prediction=model.predict(test_df)


out=prediction.round().astype(int)
print(out)
final=test_df['id']
final=pd.concat([final,pd.DataFrame(data=out,columns=['target'])],axis=1)

print(final)

final.to_csv('Results4.csv',index=False)
