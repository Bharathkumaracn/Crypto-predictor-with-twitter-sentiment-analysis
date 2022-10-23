import pandas as pd
import pandas_datareader as web
from textblob import TextBlob 
import datetime as dt 
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential

crypto_currency ="BTC"
against_currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo', start,end)



scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days=60

x_train, y_train = [],[]

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
    
x_train,y_train = np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


model = Sequential()

model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=10,batch_size=32)


test_start = dt.datetime(2021,7,1)
test_end = dt.datetime(2021,7,30)

test_data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo', test_start,test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

prediction_prices=prediction_prices[16:]
actual_prices=actual_prices[16:]

def remove_emoji(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"  
                                u"\U0001F680-\U0001F6FF"  
                                u"\U0001F1E0-\U0001F1FF"
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)

def filter_tweets(txt):
    
    txt = re.sub("@[A-Za-z0-9]+","",txt) 
    txt = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", txt)
    txt = re.sub("&","",txt)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(txt)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    filtered_sentence = ' '.join([str(elem) for elem in filtered_sentence])
    txt=filtered_sentence
    return(txt)

dateList=list()

posList=list()
negList=list()
neuList=list()


data1=pd.read_csv("C:/Users/bhara/Desktop/Bitcoin_tweets.csv",usecols=["date"], lineterminator='\n',low_memory=False)
data2=pd.read_csv("C:/Users/bhara/Desktop/Bitcoin_tweets.csv",usecols=["text"], lineterminator='\n',low_memory=False)
data1=data1["date"].tolist()
for i in range(len(data1)):
    temp=data1[i].split(" ",1)
    data1[i]=temp[0]  
    
data2=data2["text"].tolist()


c=10000
for j in range(16,31):
    textList = list()
    polarityList= list()
    sentimentList = list()
    testdate=dt.datetime(2021,7,j)
    testdate= testdate.strftime("%Y-%m-%d")
    
    for i in range(len(data1)):
        if(data1[i]==testdate):
            textList.append(data2[i])
   
    l=len(textList)
    if(l<c):
        textList=textList[:l]
    else:
        textList=textList[:c]
    
    l=len(textList)    
    for i in range(l):
        textList[i]=filter_tweets(textList[i])
        textList[i]=remove_emoji(textList[i])
        our_analysis = TextBlob(textList[i])
        polarity = our_analysis.sentiment.polarity
        
        polarityList.append(polarity)
        if(polarity >=0.1):
            sentimentList.append('Positive')
        elif(polarity <0):
            sentimentList.append('Negative')
        else:
            sentimentList.append('Neutral')
    
    dateList.append(testdate)
    print("Date: ",testdate)
    pos=0
    neg=0
    neu=0
    l=len(sentimentList)
    for i in range(l):
        if sentimentList[i]=="Positive":
            pos=pos+1
        elif(sentimentList[i]=="Negative"):
            neg=neg+1  
        else:
            neu=neu+1
    
    print('Total tweets: ',l)
    print('Positive tweets: ',pos)
    print('Negative tweets: ',neg)
    print('Neutral tweets: ',neu)
    pos_percentage=(pos/l)*100
    neg_percentage=(neg/l)*100
    neu_percentage=(neu/l)*100
    neu_percentage=round(neu_percentage,1)
    pos_percentage=round(pos_percentage,1)
    neg_percentage=round(neg_percentage,1)
    print('Positive % ',pos_percentage)
    print('Negative % ',neg_percentage)
    print('Neutral % ',neu_percentage)
    print("-----------------------------")
    posList.append(pos)
    negList.append(neg)
    neuList.append(neu)
    
ratio=[]
for i in range(len(posList)):
    ratio.append(posList[i]/negList[i])

prediction_true=prediction_prices
    
plt.plot(actual_prices[2:],color='black', label='Actual Prices')
plt.plot(prediction_prices[2:],color='green',label='Predicted Prices')
plt.title(f'{crypto_currency} Price Prediction Without Error Reduction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()



for i in range(2,len(prediction_prices)):
    dif=prediction_true[i-2]-prediction_true[i-1]
    if(prediction_prices[i-2]>prediction_prices[i-1]):
        prediction_prices[i]+=dif
    else:
        prediction_prices[i]-=dif
            
for i in range(len(ratio)):
    if(ratio[i]>1):
        prediction_prices[i]+=ratio[i]*100
    else:
        prediction_prices[i]-=ratio[i]*100
            
       
        
crypto_currency="BTC"
plt.plot(actual_prices[2:],color='black', label='Actual Prices')
plt.plot(prediction_prices[2:],color='green',label='Predicted Prices')
plt.title(f'{crypto_currency} Price Prediction After Error Correction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()