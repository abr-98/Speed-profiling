import os
import sys
import string

#sys.path.insert(0,'/') #for writing in a textfile from php

from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
file = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
import pandas as pd
import numpy as np

df=pd.read_csv(file)

X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
print(X_d)
from urllib.request import urlopen
filename='speed_generate.txt'

textfile=open(filename,"w")    #to Write in the text file
textfile.close()
l=len(X_d)
print(l)
i=0
while i<l:
    print("a")
    text=open(filename,"a")
    Wifi=X_d.iloc[i][3]
    Wifi_s=str(Wifi)
    print("b")
    Intersection=X_d.iloc[i][2]
    Intersect_s=str(Intersection)
    honk=X_d.iloc[i][0]
    honk_s=str(honk)
    print("c")

    road_surface=X_d.iloc[i][1]
    Road_surface_s=str(road_surface)
    print("d")

    timelevel=X_d.iloc[i][4]
    timelevel_s=str(timelevel)
    strng="http://localhost:5000/predict?Wifi_density="+Wifi_s+"&Intersection_density="+Intersect_s+"&Honk_duration="+honk_s+"&Timelevel="+timelevel_s+"&Road_surface="+Road_surface_s
    print(strng)

    f = urlopen(strng)

    myfile = f.read()
    print(myfile)
    var = myfile.decode('utf-8')
    text.write(var)
    text.close()
    i=i+1
import pandas as pd
import numpy as np

df=pd.read_csv(file)

y=df['Class']
y_d=pd.DataFrame(y)
print(y_d)
filename='speed_check.txt'

textfile=open(filename,"w")    #to Write in the text file
textfile.close()
l=len(X_d)
print(l)
i=0
while i<l:
    print("a")
    text=open(filename,"a")
    speed=y_d.iloc[i][0]
    speed_s=str(speed)
    print("d")
    speed_s2=speed_s+'\n'
    text.write(speed_s2)
    text.close()
    i=i+1
l=len(y_d)

textfile=open("speed_generate.txt","r")
text=open("speed_check.txt","r")
i=0
gen=[]
chk=[]
k=textfile.readlines()
l=len(k)
i=0
while i<l:

    k1=k[i].replace('\'','')
    k2=k1.replace('\n','')
    k3=k2.replace('"','')

        #print(k3)
    if k3=='Slow':
        n=1
    if k3=='Normal':
        n=2
    if k3=='Fast':
        n=3
    if k3=='Very Fast':
        n=4
    if k3=='Bad Data':
        n=0
    gen.append(n)
    i=i+1
print(gen)
#print("a")
p=text.readlines()
i=0
while i<l:
        p1=p[i].replace('\'','')
        p2=p1.replace('\n','')
        p3=p2.replace('"','')
        print(p3)
        if p3=='Slow':
            n=1
        if p3=='Normal':
            n=2
        if p3=='Fast':
            n=3
        if p3=='Very Fast':
            n=4
        if p3=='Bad Data':
            n=0
        chk.append(n)
        i=i+1
#p2=p.replace('\'',',')
#print(p)
#chk.append(p)
print(chk)
#print("b")
l=len(k)
l2=len(p)

over_cnt=0
undr_cnt=0
same_cnt=0
i=0
decision="undetermined"
while i<l:
    if gen[i]!=0 or chk[i]!=0:
        if gen[i]==chk[i]:
            decision="Accurate speeding"
            same_cnt=same_cnt+1
        if gen[i]<chk[i]:
            decision="under speed_level"
            undr_cnt=undr_cnt+1
        if gen[i]>chk[i]:
            decision="over speed_level"
            over_cnt=over_cnt+1
       # print(decision)
        i=i+1

over_s=str(over_cnt)
undr_s=str(undr_cnt)
same_s=str(same_cnt)

print("------------------DRIVER PROFILE-----------------")
print("Accurately driven segments="+same_s)
print("No of overspeeding segment="+over_s)
print("No of underspeeding segment="+undr_s)

        
