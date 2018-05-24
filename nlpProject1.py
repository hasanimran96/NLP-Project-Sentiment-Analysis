
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import string
import nltk
import matplotlib.pyplot as plt
import re
import numpy as np

from wordcloud import WordCloud
from os import path
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

origdata = pd.read_csv('Reviews.csv',
                       usecols=['Score', 'Summary', 'Text'], nrows=50000)#,skiprows=range(30000,50000))
# print(origdata.head(3))


# In[20]:


def binary(x):
    if x > 3:
        return 'positive'
    return 'negative'


Score = origdata['Score']
Score = Score.map(binary)
Text = origdata['Text']
Summary = origdata['Summary']
Xtrain, Xtest, ytrain, ytest = train_test_split(
    Text, Score, test_size=0.2, random_state=42)

rmvpunc = str.maketrans(string.punctuation, "                                ")
rmvnum = re.compile('[^a-zA-Z]+')
stemmer = SnowballStemmer("english")

def cleandata(eachText):
    eachText = rmvnum.sub(' ', eachText).strip()
    eachText = eachText.lower()
    eachText = eachText.translate(rmvpunc)
    #tokens = nltk.word_tokenize(eachText)
    #stemmed = []
    #for eachitem in tokens:
    #    stemmed.append(stemmer.stem(eachitem))
    #eachText = ' '.join(stemmed)
    return eachText


corpus = []
for eachi in Xtrain:
    corpus.append(cleandata(eachi))

countvect = CountVectorizer(min_df = 2, ngram_range = (1, 4))
Xtraincounts = countvect.fit_transform(corpus)

tfidf = TfidfTransformer()
Xtraintfidf = tfidf.fit_transform(Xtraincounts)

testset = []
for eachi in Xtest:
    testset.append(cleandata(eachi))

Xnewcounts = countvect.transform(testset)
Xtesttfidf = tfidf.transform(Xnewcounts)

df = pd.DataFrame({ 'Before': Xtrain,'After': corpus})
print(df.head(20))

prediction = dict()


# In[21]:


model = MultinomialNB(alpha=0.01).fit(Xtraintfidf, ytrain)
prediction['Multinomial'] = model.predict(Xtesttfidf)


model = BernoulliNB(alpha=0.01).fit(Xtraintfidf, ytrain)
prediction['Bernoulli'] = model.predict(Xtesttfidf)


logisticreg = linear_model.LogisticRegression(C=1e9,verbose=1)
logreg_result = logisticreg.fit(Xtraintfidf, ytrain)
prediction['Logistic'] = logisticreg.predict(Xtesttfidf)


# In[36]:


def binarynum(x):
    if x == 'positive':
        return 1
    return 0


vfunc = np.vectorize(binarynum)
fig = plt.figure(figsize=[9,9])
j = 0
colors = ['b', 'm', 'k', 'y', 'g']
for model, predicted in prediction.items():
    fprate, tprate, thresholds = roc_curve(
        ytest.map(binarynum), vfunc(predicted))
    roc_auc = auc(fprate, tprate)
    plt.plot(fprate, tprate,
             colors[j], label='%s: AUC %0.2f' % (model, roc_auc))
    j += 1


plt.legend(loc='lower right')
plt.title('Classifiers comparaison with ROC', fontsize=20)
plt.ylim([-0.1, 1.2])
plt.xlim([-0.1, 1.2])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)

#plt.show()

words = countvect.get_feature_names()
polarities = pd.DataFrame(
    data=list(zip(words, logreg_result.coef_[0])),
    columns=['feature', 'coef'])

polarities = polarities.sort_values(by='coef')

positive_words = ''

negative_words = ''

polarities_new = polarities.reset_index(drop=True)

for x in range(0, 200):
    if polarities_new['feature'][x] not in stopwords.words('english'):
        if polarities_new['feature'][x] not in negative_words:
            negative_words = negative_words + polarities_new['feature'][x].replace(" ", "_") + ' '

for x in range(len(polarities)-1, len(polarities)-199, -1):
    if polarities_new['feature'][x] not in stopwords.words('english'):
        if polarities_new['feature'][x] not in positive_words:
            positive_words = positive_words + polarities_new['feature'][x].replace(" ", "_") + ' '

good_mask = np.array(Image.open('ttup.jpg'))
bad_mask = np.array(Image.open('ttdown.jpg'))

            
wc = WordCloud(background_color="white", mask=good_mask, max_font_size=90)
# generate word cloud
wc.generate(positive_words)
fig = plt.figure(figsize=[10,10])            
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
#plt.figure()
fig.suptitle('Positive Polarity WordCloud', fontsize=40)
fig.subplots_adjust(top=2.65)
plt.show()
print('\n\n\n')
wc = WordCloud(background_color="white", mask=bad_mask, max_font_size=90)
# generate word cloud

wc.generate(negative_words.replace("ok ", "not_ok "))
fig = plt.figure(figsize=[10,10])  
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
fig.suptitle('Negative Polarity WordCloud', fontsize=40)
fig.subplots_adjust(top=2.7)
#plt.figure()
plt.show()


# In[35]:


def checkprob(model, s):
    scounts = countvect.transform([s])
    stfidf = tfidf.transform(scounts)
    result = model.predict(stfidf)[0]
    prob = model.predict_proba(stfidf)[0]
    print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))

def getFeaturePolarity(feat):
    foodlist = list()
    sum = 0
    posReviews = 0
    negReviews = 0
    for i in range(len(polarities['feature'])):
        if feat in polarities['feature'][i]:
            foodlist.append(polarities['feature'][i])
            coeff = polarities['coef'][i]
            if coeff<0:
                negReviews = negReviews + 1
            else:
                posReviews = posReviews + 1
            sum = sum+polarities['coef'][i]
    print("Feature Polarity: %s, Negative Reviews: %s , Positive Reviews: %s"%(sum,negReviews,posReviews))


while(1):
    print("\n\nChoose one of the following")
    print("0-Exit")
    print("1-Predict review sentiment")
    print("2-Check feature polarity")
    valRead = input()
    if valRead=="0":
        print("exiting...")
        break
    elif valRead=="1":
        print("Enter review")
        reviewRead = input()
        print("calculating...")
        checkprob(logisticreg,reviewRead)
    elif valRead=="2":
        print("Enter Feature")
        featRead = input()
        print("calculating...")
        getFeaturePolarity(featRead)
    else:
        print("wrong input")


# In[34]:





# In[40]:


print(ytrain.reset_index(drop=True))

