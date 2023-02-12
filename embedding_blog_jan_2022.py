import openai
from pprint import pprint
import numpy as np

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('openaiapikey.txt')

inputVal = ["feline friends go", "meow","canine companions say" , "woof", "bovine buddies say", "moo"]

resp = openai.Embedding.create(
    input=inputVal,
    model="text-embedding-ada-002")
    #model="text-similarity-ada-001")
    #model="text-similarity-davinci-001")
    
resultList = list()
for index, val in enumerate(inputVal):
    if index%2 == 0:
            for x in range(1,6,2):
                resultVal = inputVal[index] + ' and ' + inputVal[x] + ' is ' + str(np.dot(resp['data'][index]['embedding'], resp['data'][x]['embedding']))
                resultList.append(resultVal)
    
pprint(resultList)
