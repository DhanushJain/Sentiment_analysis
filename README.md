# Sentiment Analysis
Headline scraping from given website and using that in sentiment analysis built using sequential nn model. 
## Run
Dataset I have used for training in this model is from "Million News Headlines", the dataset was downloaded from [kaggle](https://www.kaggle.com/datasets/therohk/million-headlines?resource=download).
I am using google colab to run my code, it can be downloaded and ran on notebook too.
Make sure all the modules in requirements.txt is met, then you can run the code on your own.
## JSON file format
The JSON file consists of key as numbers ranging till number of headlines and in it with headline as the key, headline text is attached to it.

```json
{"0": {"Headline": "'Floods hit South Africa's KwaZulu-Natal province again'"}, "1": {"Headline": "'Mozambique: Cyclone Gombe death toll rises to 53'"}, "2": {"Headline": "'Mozambique announces new prime minister after cabinet reshuffle'"}, "3": {"Headline": "'Analysis: Can African gas replace Russian supplies to Europe?'"}, "4": {"Headline": "'Dozens dead from Tropical Storm Ana in southern Africa'"}, "5": {"Headline": "'Southern Africa bloc SADC extends Mozambique mission'"}, "6": {"Headline": "'Climate change and famine | Start Here'"}, "7": {"Headline": "'In Mozambique, Kagame says Rwandan troops' work not over'"}, "8": {"Headline": "'Rwanda, Mozambique forces recapture port city from rebels'"}, "9": {"Headline": "'Rwanda deploys 1,000 soldiers to Mozambique's Cabo Delgado'"}}
```
## Web Scraping
The module I use for web scraping is BeautifulSoup, as it provides all ways to edit, extract and filter contents as desired.  
Here, the content is taken by the request from the website link, and fed into BeautifulSoup and is parsed as HTML, to keep the extracted headlines in JSON format it is stored in dict.  
As the text I retrieved from the website contains \xad, the text was converted into ascii to replace the text and then the ascii values variables are converted back to ascii values, as there is no direct way to do it, I had to do it manually.
```python
def get_headlines(headline_count=10):
    content = requests.get('https://www.aljazeera.com/where/mozambique/')
    soup = BeautifulSoup(content.content, 'html.parser')
    # to keep the data in json format
    top_headlines = {}
    all_a = soup.find_all('article')
    for i in range(headline_count):
        headline = all_a[i].find('h3')
        text = headline.get_text().strip()
        text = ascii(text).replace('\\xad','')
        # to get back the ascii values 
        text = text.replace('\\n','')
        text = text.replace('\\u2019',"'")
        top_headlines.update({i: {'Headline': text}})
    return top_headlines
```
## Sequential NN model
The library used to run this model is keras, this is a high level NN library which runs on top of TensorFlow.  
The pre-processed data is taken and then split into test and training data with their labels.
```python
text = list(data['text'])
labels = list(data['label'])
# training and test data of text
training_text = text[0:20000]
testing_text = text[20000:]
# training and test data of labels
training_labels = labels[0:20000]
testing_labels = labels[20000:]
```
Then the tokenizer is used for word encodings in dictionary which is taken from keras library, the texts in the data are then converted into sequences and the sequences which need padding are then padded using pad_sequence instance.
```python
tokenizer = Tokenizer(num_words=10000, oov_token= "<OOV>")
tokenizer.fit_on_texts(training_text)

word_index = tokenizer.word_index
#sequencing and padding
training_sequences = tokenizer.texts_to_sequences(training_text)
training_padded = pad_sequences(training_sequences, maxlen=100, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_text)
testing_padded = pad_sequences(testing_sequences, maxlen=100, padding='post', truncating='post')
# TensorFlow input is in np array 
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```
To create layers for the NN, first the embedding is created with input parameters such as vocab size, embedding dim and input length.
I used maxpooling in 1D to reduce the dimentions and RelU is used to provide the negative and positive classification for the model, and the sigmoid activation function helps the model to provide a probability of the headline being negative or positive in the range of [0,1]. The optimizer using to reduce the loss and fit the curve is adam, and binary entropy is the function used to calculate the model loss.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```
The model summary shows the number of param, and embedding dim of the sequential model.
```python
model.summary()
```
```txt
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           160000    
                                                                 
 global_average_pooling1d (G  (None, 16)               0         
 lobalAveragePooling1D)                                          
                                                                 
 dense (Dense)               (None, 24)                408       
                                                                 
 dense_1 (Dense)             (None, 1)                 25        
                                                                 
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0
_________________________________________________________________
```
The model is ran with 15 epochs, you can increase the epochs to fit the model better, but be careful of overfitting.
```python
num_epochs = 15
history = model.fit(training_padded, 
                    training_labels, 
                    epochs=num_epochs, 
                    validation_data=(testing_padded, testing_labels), 
                    verbose=2)
```
The epochs shows the accuracy as 96.47% with very negligible loss in the model.
```txt
Epoch 1/15
625/625 - 4s - loss: 0.6356 - accuracy: 0.6332 - val_loss: 0.5364 - val_accuracy: 0.6886 - 4s/epoch - 6ms/step
Epoch 2/15
625/625 - 3s - loss: 0.3306 - accuracy: 0.8854 - val_loss: 0.2338 - val_accuracy: 0.9167 - 3s/epoch - 4ms/step
Epoch 3/15
625/625 - 2s - loss: 0.1522 - accuracy: 0.9553 - val_loss: 0.1500 - val_accuracy: 0.9509 - 2s/epoch - 4ms/step
Epoch 4/15
625/625 - 2s - loss: 0.0958 - accuracy: 0.9722 - val_loss: 0.1275 - val_accuracy: 0.9544 - 2s/epoch - 4ms/step
Epoch 5/15
625/625 - 3s - loss: 0.0678 - accuracy: 0.9804 - val_loss: 0.1067 - val_accuracy: 0.9611 - 3s/epoch - 4ms/step
Epoch 6/15
625/625 - 3s - loss: 0.0497 - accuracy: 0.9855 - val_loss: 0.0947 - val_accuracy: 0.9653 - 3s/epoch - 4ms/step
Epoch 7/15
625/625 - 3s - loss: 0.0383 - accuracy: 0.9898 - val_loss: 0.0936 - val_accuracy: 0.9665 - 3s/epoch - 4ms/step
Epoch 8/15
625/625 - 3s - loss: 0.0297 - accuracy: 0.9922 - val_loss: 0.0873 - val_accuracy: 0.9675 - 3s/epoch - 4ms/step
Epoch 9/15
625/625 - 3s - loss: 0.0232 - accuracy: 0.9941 - val_loss: 0.0860 - val_accuracy: 0.9678 - 3s/epoch - 4ms/step
Epoch 10/15
625/625 - 3s - loss: 0.0184 - accuracy: 0.9953 - val_loss: 0.0867 - val_accuracy: 0.9682 - 3s/epoch - 5ms/step
Epoch 11/15
625/625 - 3s - loss: 0.0149 - accuracy: 0.9965 - val_loss: 0.0985 - val_accuracy: 0.9659 - 3s/epoch - 4ms/step
Epoch 12/15
625/625 - 2s - loss: 0.0115 - accuracy: 0.9975 - val_loss: 0.1032 - val_accuracy: 0.9644 - 2s/epoch - 4ms/step
Epoch 13/15
625/625 - 3s - loss: 0.0095 - accuracy: 0.9982 - val_loss: 0.1053 - val_accuracy: 0.9639 - 3s/epoch - 4ms/step
Epoch 14/15
625/625 - 2s - loss: 0.0079 - accuracy: 0.9980 - val_loss: 0.0954 - val_accuracy: 0.9671 - 2s/epoch - 4ms/step
Epoch 15/15
625/625 - 2s - loss: 0.0056 - accuracy: 0.9991 - val_loss: 0.1044 - val_accuracy: 0.9647 - 2s/epoch - 4ms/step
```
## Plot
The plot is performed by using plotly module, as it provides the best functions to work around with.
```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1,len(Analysis_results)+1)),
    y=Analysis_results
))

fig.update_layout(
    autosize=False,
    width=700,
    height=500,
    paper_bgcolor='lightgrey'
)

# showing the plot
fig.show()
```
![./Sentiment plot.png](https://github.com/DhanushJain/Sentiment_analysis/blob/main/Sentiment%20plot.png))
## Why sequential NN model approach
Sequential model is used when the input tensors is one and output tensors is one.  
With the use of just machine learning algorithms, the parameters considered to perform the analysis and training is less.  
Building model using naive bayes classifer yeilded with less accuracy, which would inturn affect the prediction of the model.
As the model does not have non-linear topology such as residual connections, this sequential model works best with this model.  
The execution time for this model was consiterablely lower than when it was ran on naive bayes classifier, this will help when rebuilding the model if necessary faster and efficiently.

