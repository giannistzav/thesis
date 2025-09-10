import pandas as pd
import pickle
filename='sentences-5words.txt'
data = pd.read_csv("sentences.txt",sep="\r", header = None)
data = data.rename(columns={0: 'sentences'})
sent=[]
for i in range (len(data)):
    temp=data['sentences'][i].split()
    temp=temp[0:5]
    #data.loc[i,'sentences'] = ' '.join(temp)
    sent.append(' '.join(temp))
    
with open(filename, 'wb') as f:
        pickle.dump(sent, f)



