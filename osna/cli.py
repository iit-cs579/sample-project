# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

from . import credentials_path, clf_path

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('collect')
@click.argument('directory', type=click.Path(exists=True))
def collect(directory):
    """
    Collect data and store in given directory.
    """
    pass


@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics of your approach.
    """
    pass


@main.command('network')
def network():
    """
    Perform the network analysis component of your project.
    """
    pass

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all data and print statistics.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it for later use in the web
    """
    print('reading from %s' % directory)
    # (1) Read the data...
    #
    # (2) Create classifier and vectorizer.
    clf = LogisticRegression() # set best parameters 
    vec = CountVectorizer()    # set best parameters

    # save the classifier
    pickle.dump((clf, vec), open(clf_path, 'wb'))



@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    """
    Launch a web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
    


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
