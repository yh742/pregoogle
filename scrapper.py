#!/usr/bin/env python

import requests
import sys
import bs4
import csv
import time

URL = "http://allrecipes.com/recipe/getreviews/?recipeid=%s&pagenumber=%s&pagesize=1000&recipeType=Recipe&sortBy=MostPositive"

if __name__ == "__main__":
    print sys.argv
    if len(sys.argv) < 2:
        exit()
    lines = []
    #with open("no_safe_links.txt", "r") as f:
    with open("safe_links.txt", "r") as f:
        lines = f.readlines()
    ids = [x.split('/')[4] for x in lines]

    #f = open("no_safe_recipe.csv", 'wb')
    f = open("no_safe_recipe.csv", 'wb')
    writer = csv.writer(f)
    for id in ids:
        pageNum = 1
        while True:
            url = URL % (id, pageNum)
            print "AllRecipe URL: " + url
            req = requests.get(url)
            blah = req.text.strip()
            if blah == "":
                break
            soup = bs4.BeautifulSoup(req.text, "html5lib")
            metas = soup.find_all("meta")
            ratings = []
            for meta in metas:
                if meta.has_attr("itemprop") and meta["itemprop"] == "ratingValue":
                    ratings.append(meta["content"])
            review_dates = [x.text for x in soup.find_all(class_="review-date")]
            reviews = [x.text for x in soup.find_all("p", {"itemprop": "reviewBody"})]
            print ratings
            for x in range(len(ratings)):
                writer.writerow([review_dates[x], ratings[x], reviews[x].replace('"', "").replace(",","").strip().encode('utf-8')])
            pageNum += 1
        time.sleep(3)
    f.close()