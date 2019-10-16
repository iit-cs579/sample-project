# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.
import click
import json
import glob
import pickle
import sys

import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

from . import credentials_path, clf_path
from .mytwitter import Twitter

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('collect')
@click.argument('directory', type=click.Path(exists=True))
def collect(directory):
    """
    Collect data and store in given directory.

    This should collect any data needed to train and evaluate your approach.
    This may be a long-running job (e.g., maybe you run this command and let it go for a week).
    """
    twitter = Twitter(credentials_path)
    limit = 100
    fname = directory + os.path.sep + 'data.json'
    outf = open(fname, 'wt')
    ncollected = 0
    for tw in twitter.request('statuses/filter', {'track': 'chicago'}):
        print(tw['text'])
        outf.write(json.dumps(tw, ensure_ascii=False) + '\n')
        ncollected += 1
        if ncollected >= limit:
            break
    print('collected %d tweets, stored to %s' % (ncollected, fname))
    outf.close()

@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics of your approach.
    For example, compare classification accuracy for different
    methods.
    """
    pass


@main.command('network')
def network():
    """
    Perform the network analysis component of your project.
    E.g., compute network statistics, perform clustering
    or link prediction, etc.
    """
    pass

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all data and print statistics.
    E.g., how many messages/users, time range, number of terms/tokens, etc.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier on all of your labeled data and save it for later
    use in the web app. You should use the pickle library to read/write
    Python objects to files. You should also reference the `clf_path`
    variable, defined in __init__.py, to locate the file.
    """
    print('reading from %s' % directory)
    # (1) Read the data...
    #
    # (2) Create classifier and vectorizer.
    # You can use any classifier you like.
    clf = LogisticRegression() # set best parameters 
    vec = CountVectorizer()    # set best parameters

    # save the classifier
    pickle.dump((clf, vec), open(clf_path, 'wb'))



@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
    


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
