from textblob.classifiers import NaiveBayesClassifier
import random
import textblob

safe_stuff = []
no_safe_stuff = []

def clean(txt):
    not_list = ['...', 'you', 'your', 'his', 'her', 'will', "didn't", "i'll"]
    replace_list = [".", "!", "#", "\r\n", "-", "@", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "$"]
    cleaned = []
    for token in txt.split(' '):
        flag = False
        for y in not_list:
            if y in token.lower():
                flag = True
                break
        if flag: continue
        for y in replace_list:
            token = token.replace(y, "")
        if len(token) > 3:
            token = token.strip().lower();
            try:
                word = textblob.TextBlob(token).words[0]
                cleaned.append(word.lemma)
            except:
                continue
    return ' '.join(cleaned).decode("ascii", "ignore")

#with open("no_safe_recipe.csv", "rb") as f:
with open("safe_recipe.csv", "rb") as f:
    for x in f:
        tokens = x.split(",")
        if len(tokens) > 2:
            #print tokens[2]
            safe_stuff.append((clean(tokens[2]), "neg"))

with open("no_safe_recipe.csv", "rb") as f:
    for x in f:
        tokens = x.split(",")
        if len(tokens) > 2:
            no_safe_stuff.append((clean(tokens[2]), "neg"))

x  = int(0.3*len(safe_stuff))
y =  int(0.3*len(no_safe_stuff))

print safe_stuff[:10]
print no_safe_stuff[:10]

random.shuffle(safe_stuff)
random.shuffle(no_safe_stuff)
safe_train = safe_stuff[:x]
no_safe_train = no_safe_stuff[:y]

safe_test = safe_stuff[x:x+2000]
no_safe_test = no_safe_stuff[y:y+2000]

cl = NaiveBayesClassifier(safe_train + no_safe_train)
