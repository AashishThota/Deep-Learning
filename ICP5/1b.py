from keras.models import load_model

model=load_model('lstm.h5')
sentence="evil always wins"
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(sentence)
X = tokenizer.texts_to_sequences(sentence)
#print(X)
X = pad_sequences(X,maxlen=28)
sentiment = model.predict(X,batch_size=1,verbose = 2)[0]

print(sentiment)