import re
import math
from collections import Counter
def removepunctuation(string):
  a=[',',"'",":",'"',")","(","?","!","-","\n",",","."]
  string=string.lower()
  string = re.sub(r'\s+', ' ', string).strip()
  for i in string:
    if i in a:
      string=string.replace(i," ")
  if string[0]==' ':
    string=string[1:]
  if string[-1]==' ':
    string=string[:-1]
  return string


def unique(pieces):#returns dictionary
  return set(list(pieces))



def cutintopieces(string,n):#returns string
  string=string.lower()
  pieces=[]
  for i in range(0,len(string),n-1):
    if i+n>len(string):
      break
    a=string[i:i+n]
    pieces.append(a)
  return pieces



def entropy(string):
  entropy=0
  string=removepunctuation(string)
  b=cutintopieces(string,3) #n-grams
  c=Counter(b)
  for i in c:
    entropy+=-(c[i]/len(c))*math.log(c[i]/len(c),2)
  return entropy



def ttr(string):#type-token-ratio
    a=len(unique(string))
    b=len(string)
    return a/b