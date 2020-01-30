---
layout: post
title: An introduction to web scraping and sentiment analysis.
description: >
  An introduction to web scraping, sentiment analysis and how text/web data analysis can be useful in advertising, marketing and 
  customer service. 
noindex: true
comments: true
---
Welcome back to the blog folks! Sorry for the long break! It's been a relatively busy holiday season for me, so the blog has taken a back seat for a bit. However, I'm back with more time and ideas, so expect a more frequency for the next few posts. Today, we're going to be giving a short intro to both web scraping and easy to understand and implement sentiment analysis algorithms (like [`VADER`](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)). In particular, I've scraped 4 websites over an ~4 year history for articles that are particularly rich in holiday season buzzwords and phrases (think 'Christmas', 'New Year', 'holiday', 'cheer', etc.). For every article, I score the polarity of the text using the `VADER` algorithm, record the counts of whatever holiday phrase is being detected, the article date and a few other items. I then analyze the recorded data as time series to observe how the counts and scores change over the seasons of the 4 years. If this brief description sounds interesting, but maybe isn't totally clear, then enjoy reading!

## What is webscraping?
Okay, first thing's first! Let's tackle the concept of web scraping. In general, web scraping is data scraping used for extracting data from websites. Typically, specific data is gathered and copied from the web into local storage for later retrieval or analysis. Scraping can be done manually, but is often automated by a simple bot called a web crawler. Part of today's blog will be showing you all how to make a simple web crawler for RSS feeds using `Python 3` and some of its packages (namely `requests`, `bs4`, `feedparser` and `re`). Let's start with making the necessary imports.

~~~python
import requests
import bs4
from bs4.element import Comment
import feedparser
import re

from htmldate import find_date
import csv

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
~~~

Let's take a look through the packages above and discuss what they're for. Before, I named 4 packages that were most important for this web scraper build: [`requests`](https://requests.readthedocs.io/en/master/), [`bs4`](https://beautiful-soup-4.readthedocs.io/en/latest/), [`feedparser`](https://github.com/kurtmckee/feedparser) and [`re`](https://docs.python.org/3/library/re.html). `requests` allows you to se HTTP requests and connect with sites. `bs4` is a library for  pulling data out of HTML and XML files; it allows us to parse the information from the websites we're scraping. `feedparser` allows us to parse ATOM and RSS feeds. Finally, `re` is a native `Python` library for regular expression operations that allows us to 'clean up' the text we get from a scrape by getting rid of delimiters, punctuations, etc. Some of the other imports you see are for storage or other analyses that we're doing *in vivo* with the scraping. [`htmldate`](https://htmldate.readthedocs.io/en/latest/) is a great, easy-to-use library for finding the publication date of web pages. [`csv`](https://docs.python.org/3/library/csv.html) is just the native `Python` package for reading and writing csv (which I chose to use for this for simplicity familiarity, though XML or JSON might be a better choice). [`nltk`](https://www.nltk.org/) is `Python`'s natural language tool kit and processing library. It is very versatile and powerful, but today I'm just using it for the `VADER` implementation. Now that we have an idea of what everything is for, let's move on to building the crawler. But first...

### A brief aside on the `VADER` algorithm
`VADER` (or Valence Aware Dictionary for sEntiment Reasoning) is a simple, rule-based model for general sentiment analysis. 

We'll go piece by piece explaining what each function of the crawler is meant to do. First, I want to build a masking function that filters out any text that isn't the body of the article we're analyzing. That could be metadata, comments, ad information, pictures, etc.

~~~python
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True
~~~

Pretty much, I've provided common HTML parent element names that I don't want and also the `bs4` built-in class for detecting comments sections and if the text is any of those things, they get thrown out. Next, I want to start by building the function that actually searches my website for whatever holiday-themed string I've specified.

~~~python
def search_article(url, phrases):
    """
    Yield all of the specified phrases that occur in the HTML body of the URL.
    """
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts) 
    visible_text = u" ".join(t.strip() for t in visible_texts)
    
    for phrase in phrases:
        if re.search(r'\b' + re.escape(phrase) + r'\b', visible_text):
            words = re.findall(r'\b' + re.escape(phrase) + r'\b', visible_text)
            date = find_date(response.text)
            polarity_score = sia.polarity_scores(visible_text)['compound']
            yield phrase, len(words), date, polarity_score
~~~

First we connect using `request`, parse the HTML using `bs4` and pass it through our article body text filter. Next, in a for-if loop, we parse our text using `re`, searching for our holiday phrases, getting a count of phrase per article, an article date and calculating the polarity of the article with VADER. Next, we need a function to call each article in a RSS feed an return the scraped info I want. 

~~~python
def search_rss(rss_entries, phrases):
    """
    Search articles listed in the RSS entries for phrases, yielding
    (url, article_title, phrase, phrase_count, datetime, vader_score) tuples.
    """
    for entry in rss_entries:
        for hit_phrase, number, datetime, score in search_article(entry['link'], phrases):
            yield entry['link'], entry['title'], hit_phrase, number, datetime, score
~~~

Connecting all of the parts, we have our `main()` function (which is like a standardized point of execution).

~~~python
def main(rss_url, phrases, output_csv_path, rss_limit=None):
    rss_entries = feedparser.parse(rss_url).entries[:rss_limit]
 
    with open(output_csv_path, 'a') as f:
        w = csv.writer(f, delimiter=';', quotechar='"')
        for url, title, phrase, number, datetime, score in search_rss(rss_entries, phrases):
            print('"{0}" found {1} times in "{2}" on "{3}". Overall polarity score: "{4}"'.format(phrase, number, title, datetime, score))
            w.writerow([phrase, number, title, datetime, url, score])
~~~

Given an RSS feed URL, list of phrases, and output path, we parse the RSS feed for URL entry links and then analyze each linked article for phrase counts, datetime and VADER score, recording each URL-phrase combination as its own row of a CSV. Putting all the code snippets together and placing a top-level script execution for `main()` where I specify the primary RSS sites, the date ranges I'm interested in and the holiday-phrases I want to detect, we have a fully-functioning, automated web crawler!

~~~python
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def search_article(url, phrases):
    """
    Yield all of the specified phrases that occur in the HTML body of the URL.
    """
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts) 
    visible_text = u" ".join(t.strip() for t in visible_texts)
    
    for phrase in phrases:
        if re.search(r'\b' + re.escape(phrase) + r'\b', visible_text):
            words = re.findall(r'\b' + re.escape(phrase) + r'\b', visible_text)
            date = find_date(response.text)
            polarity_score = sia.polarity_scores(visible_text)['compound']
            yield phrase, len(words), date, polarity_score

def search_rss(rss_entries, phrases):
    """
    Search articles listed in the RSS entries for phases, yielding
    (url, article_title, phrase) tuples.
    """
    for entry in rss_entries:
        for hit_phrase, number, datetime, score in search_article(entry['link'], phrases):
            yield entry['link'], entry['title'], hit_phrase, number, datetime, score

def main(rss_url, phrases, output_csv_path, rss_limit=None):
    rss_entries = feedparser.parse(rss_url).entries[:rss_limit]
 
    with open(output_csv_path, 'a') as f:
        w = csv.writer(f, delimiter=';', quotechar='"')
        for url, title, phrase, number, datetime, score in search_rss(rss_entries, phrases):
            print('"{0}" found {1} times in "{2}" on "{3}". Overall polarity score: "{4}"'.format(phrase, number, title, datetime, score))
            w.writerow([phrase, number, title, datetime, url, score])

if __name__ == '__main__':
    rss_url_base =  ['http://rss.cnn.com/rss/cnn_latest.rss',
                     'http://feeds.washingtonpost.com/rss/rss_blogpost',
                     'https://www.wired.com/feed',
                     'http://feeds.reuters.com/reuters/entertainment']
    
    year = ['2019','2018','2017','2016']
    month = ['12','11','10','09','08','07','06','05','04','03','02','01']
    day = ['28','21','14','07','01']
    
    rss_wayback_archives = ['https://web.archive.org/web/'+yr+mm+dd+'/'+base for yr in year for mm in month for dd in day for base in rss_url_base]
          
    phrases = ['holiday', 'cheer', 'give', 'giving', 'family' 'happy', 'gift', 'season', 'tis', 'Jesus',
               'Christ', 'miracle', 'birth' 'merry', 'new', 'year', 'happy holidays', 'new year', 
               'Christmas', 'Happy Holidays', 'New Year']
    
    for url in rss_wayback_archives:
        main(url, phrases, 'output_filtered_long.csv', 10000)
~~~

There are some important things to note about the crawler structure that might be confusing. First, quite a few of my functions use `yield` instead of `return`: the main reason for that is `yield` allows a function to stop executing, send a value back to its caller, but keep enough state to resume where it left off. This allowed me to write the CSV in real time (one article analysis at a time) instead of returning all of my information at the end of the scraping and write to file. This is easier on memory and is a nice feature in case a request fails or there's a hangup (I can start where the CSV stopped recording). Second, I specify a date range for analysis and feed it into URLs all connected to [The Wayback Machine web archive](https://archive.org/web/). RSS feeds aren't typically very long and companies/news sites frequently update them with 25-50 latest entries. Hence, in order to do a historical scraping, I chose 4 days out of each month from the past 4 years and went to the RSS feeds stored by The Wayback Machine on those dates to scrape. This seemed like a fair tradeoff between granularity of getting all possible articles and redundancy in having repeat articles between feed snapshots on consecutive days. Once we run the crawler and waiting about a day (give or take), we end up with ~25k rows of scraped article data :) !



![Overlay KDEs](/assets/img/blog4/resample_dist_overlays.png)


If we look more closely at our shuffle split and bootstrap distributions (which are probably the most reliable), they look very similar (unsuprisingly). If I were to increase the number of repetitions in the shuffle split algorithm, it would probably very closely resemble the bootstrap distribution.

![Overlay KDEs B&SP](/assets/img/blog4/resample_dist_overlays_just_Shuffsplit_Bootstrap.png)


## Conclusions
There are many potential resampling methods for estimating uncertainty in a model. These are some good, generalizable ones that can be easily implemented for machine learning algorithms. The bootstrap is likely the most robust and general and is my go-to quick method for uncertainty estimation. It might get its own post in the future because it is very useful! Hope you learned and enjoyed! Until next time, folks. 
