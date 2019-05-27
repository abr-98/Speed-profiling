import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
import flask
import numpy as np
#import tensorflow as tf
from keras.models import load_model
from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle


df=pd.read_csv('6mar.csv')


model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.1,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)

    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])

    else:

        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1

#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)
model.save('model/speed_detect.h5')



app = flask.Flask(__name__)
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('model/speed_detect.h5')
    graph = tf.get_default_graph()

def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('Wifi_density'))
    parameters.append(flask.request.args.get('Intersection_density'))
    parameters.append(flask.request.args.get('Honk_duration'))
    parameters.append(flask.request.args.get('Timelevel'))
    parameters.append(flask.request.args.get('Road_surface'))

    return parameters

def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

@app.route("/predict", methods=["GET"])
def predict():

    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1,5)
    with graph.as_default():
        raw_prediction = model.predict(inputFeature)
    print(raw_prediction)

   # k_1=round(raw_prediction[0])
    k_1_i=int(raw_prediction[0][0])
        #k_1_s=str(k_1_i)
    #k_2=round(raw_prediction[1])
    k_2_i=int(raw_prediction[0][1])
        #k_2_s=str(k_2_i)
    #k_3=round(raw_prediction[2])
    k_3_i=int(raw_prediction[0][2])
        #k_3_s=str(k_3_i)
   # k_4=round(raw_prediction[3])
    k_4_i=int(raw_prediction[0][3])
        #k_4_s=str(k_4_i)

    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        prediction='Bad Data'

    else:
        if k_1_i==1:
            prediction='Fast'
        if k_2_i==1:
            prediction='Normal'
        if k_3_i==1:
            prediction='Slow'
        if k_4_i==1:
            prediction='Very Fast'
    print(prediction)

    return sendResponse(prediction)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)
