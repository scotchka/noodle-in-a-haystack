import fuzzy, pickle
import numpy as np
import re
from flask import json, Flask, request, Response
app = Flask(__name__)

cpt = pickle.load(open('cpt.pickle'))
vectorizer = pickle.load(open('vectorizer.pickle'))
pos_all = pickle.load(open('pos.pickle'))

def sound_like(s):
    '''
    Given string of words, returns string of sound equivalents.
    '''
    sound = fuzzy.nysiis #pick algorithm
    word_list =  s.split()
    sound_list = []
    for word in word_list:
        if sound(word):
            sound_list.append(sound(word))
    return ' '.join(sound_list)

def fuzzy_items(items):
    '''
    Given list of menu items, return list of sound encoded items
    '''
    return [sound_like(item) for item in items if sound_like(item)]

def rating(p,pos):
    if p < pos.mean() - pos.std():
        return '***'
    elif p < pos.mean() + pos.std():
        return '**'
    else:
        return '*'

def rate_menu(menu):
    '''
    Given menu as string, each item on separate line, returns rated menu as string.
    '''
    items = [item.strip() for item in re.split(r'\n\W+\n',menu) if item.strip()]
    
    items_sound = fuzzy_items(items)
    
    vectors = vectorizer.transform(items_sound)
    
    for ij in zip(vectors.nonzero()[0], vectors.nonzero()[1]):
        vectors[ij] = 1
    
    vectors = np.matrix(vectors.toarray()).transpose()
    
    penalty = []
    for item in items_sound:
        p = 0
        for word in item.lower().split():
            if word.strip() not in vectorizer.get_feature_names():
                p += 1
        penalty.append(p)
    penalty = np.matrix(penalty)
    
    word_count = np.matrix([len(item.split()) for item in items_sound])
    
    pos = (cpt*vectors + penalty*-10)/word_count
    
    rated_items = []
    for item in zip(pos.tolist()[0], items):
        rated_items.append(rating(item[0],pos) + '\t' + item[1][:80])
        
    return 'Rating System: more stars = more adventurous!\n\n' + \
        '\n'.join(rated_items)

@app.route('/rate', methods = ['POST'])
def api_message():
    menu = request.form['menu']
    result = rate_menu(menu)
    resp = Response(result, status=200, mimetype='text/plain')
    return resp

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5001)