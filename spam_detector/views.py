from django.shortcuts import render, redirect
from django.contrib import messages

import pickle
import os
from nltk.stem import PorterStemmer

models_path = os.path.join(os.getcwd(), 'spam_detector/models')
tv = pickle.load(open(os.path.join(models_path, 'CountVectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(models_path, 'spam_classifier.pkl'), 'rb'))

# def process_message(message):


# Create your views here.
def index(request):
    if request.method == 'POST':
        message = request.POST['message']
        message = ' '.join([PorterStemmer().stem(word) for word in message.split()])
        if len(message) == 0:
            messages.error(request, "Please enter some message.")
            return redirect('index')
        else:
            message = tv.transform([message]).toarray()
            message = message[0].reshape((1,-1))
            print(model.predict(message))
            if model.predict(message):
                messages.error(request, "It's a SPAM Message.")
                return redirect('index')
            else:
                messages.success(request, "It's not a SPAM Message.")
                return redirect('index')   
    return render(request, 'spam_detector/index.html')
