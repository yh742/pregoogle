from flask import Flask, render_template
from flask import send_from_directory
import algo
from flask import request
import requests
import sys, pandas
import bs4
import csv
import time

app = Flask(__name__)

URL = "http://allrecipes.com/recipe/getreviews/?recipeid=%s&pagenumber=%s&pagesize=100&recipeType=Recipe&sortBy=MostPositive"

def remove_txt(txt_string):
    return ''.join([i if ord(i) < 128 else ' ' for i in txt_string])

def AllRecipe_Lookup(id):
    pageNum = 0
    #while True:
    url = URL % (id, pageNum)
    req = requests.get(url)
    blah = req.text.strip()
    if blah == "":
        return None
    soup = bs4.BeautifulSoup(req.text, "html5lib")
    meta = soup.find("meta")
    recipe = ""
    if meta.has_attr("content"):
        recipe = meta["content"]
    else:
        recipe = id
    reviews = [x.text for x in soup.find_all("p", {"itemprop": "reviewBody"})]
    return recipe, [remove_txt(reviews[x].replace('"', "").replace(",","").strip()) for x in range(len(reviews))]

@app.route("/")
def hello():
    return send_from_directory('static', 'index.html')

@app.route("/prego", methods=['GET', 'POST'])
def prego():
    term = request.form.get('search_term')
    name, lookups = AllRecipe_Lookup(term)
    safe = algo.get_prediction2(lookups)
    if safe == "not_safe":
        safe = "PreGoogle Predicts: Not Safe"
    else:
        safe = "PreGoogle Predicts: Safe"
    return render_template('prego.html', user_input=name, safe_or_not=safe)


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
