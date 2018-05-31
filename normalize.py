import pickle
f=open("data")
x=pickle.load(f)
y=list()
for i in x:
    #s=sum(i[0])
    temp=list()
    for j in i[0]:
        temp.append(float(j/255))
    y.append([temp,i[1]])

g=open("data1","w")
pickle.dump(y,g)
g.close()
