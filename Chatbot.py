import nltk
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()#make raw data be able to read
raw = raw.lower()# converts to lowercase

#Tokenization(covert string value to numerical value)
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

#Preprocess (cleaning data using natural language toolkit)
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
#Normalization
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","yo","wassup","good morning", "good afternoon", "good evening")
GREETING_RESPONSES = ["Hello there", "hey", "*nods*", "hi there", "hello", "yes i'm here to help you, How may i assist you today?"]
#Greeting words that you can use to start a conversation
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        

#Generating chatbot response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you question"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

#chatbot introduction after running the code
flag=True
print("FitBot: My name is FitBot. I will answer your questions. If you want to exit, say Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='ok' or user_response=='thank you' or user_response=='i got it'):
            flag=False
            print("FitBot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("FitBot: "+greeting(user_response))
            else:
                print("FitBot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("FitBot: Bye! take care..")    