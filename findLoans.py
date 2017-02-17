import sqlite3
import re, json, scrape, time

from collections import defaultdict
from itertools import groupby
from statistics import mean, median, variance
from math import log
#from scipy.stats import ttest_ind, ks_2samp, levene, ansari

import matplotlib.pyplot as plt
import numpy

from sklearn.cluster import MeanShift, DBSCAN
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import apiTools
from configparser import ConfigParser

con = sqlite3.connect('current.db')
cur = con.cursor()

hcon = sqlite3.connect('history.db')
hcur = hcon.cursor()


import warnings
warnings.filterwarnings("ignore")

with open("hazards.json", "r") as f:
    hazards = json.load(f)

config = ConfigParser()
config.read('config.ini')

def processScores(inlist):
    cleanScores = [x[0].replace('"','') for x in inlist]
    splitScores = [int(x.split("-")[0]) for x in cleanScores]

    return splitScores
    
def calcPayment(pv, interest, periods):
    rate = float(interest) / 1200

    payment = (rate * pv) / (1-(1+rate) ** (-1.0 * float(periods)))
    
    return payment
    
def amortTable(pv, payment, interest, periods):
    rate = float(interest) / 1200
    
    tableVals = []
    for i in range(int(periods)):
        #print(pv)
        owedInterest = pv * rate

        if pv + owedInterest < payment:
            payment = pv + owedInterest
            
        appliedToPrincipal = payment - owedInterest
        pv -= appliedToPrincipal
        
        tableVals.append((owedInterest, pv, payment))
        
    return tableVals

def calcIRR(loanVals):
    pv = -1.0 * loanVals['askPrice']
    
    monthlyPayment = calcPayment(loanVals['originalNote'], loanVals['rate'] / 100, loanVals['loanTerm'])
    aTable = amortTable(loanVals['principal'], monthlyPayment, loanVals['rate'] / 100, loanVals['remainingPmts'])

    vals = []
    for p, h in zip(reversed(aTable), reversed(hazards[loanVals['loanClass']])):
        vals.append(p[2] * h)

    rate = numpy.irr([pv, *vals])
    
    return rate * 100
     
def findLoans(price):
    fields = [  'LoanId', 'NoteId', 'OrderId','OutstandingPrincipal', 'CreditScoreTrend','FICO',
                'LoanClass', 'LoanMaturity','OriginalNoteAmount', 'InterestRate', 'RemainingPayments', 'AskPrice']

    print("Loading Data...")
    cur.execute('select {} from loans where status = "Current" and AskPrice < {} and Markup < 0'.format(",".join(fields), price))

    hcur.execute('select loanClass, avg(pmtHistoryScore) from loans group by loanClass')
    avgScores = {x[0]:x[1] for x in hcur.fetchall()}
    
    ids = []
    xVals = []
    yVals = []

    for row in cur.fetchall():
        ids.append(list(row[:3]))
        xVals.append(list([x for x in row[3:-1]]))
        yVals.append(float(row[-1]))
        
    print("Cleaning Variables...")
    # Clean up variables
    delIdx = []
    
    cstIdx = 1
    loanClassIdx = 3
    ficoIdx = 2
    
    for idx in range(len(xVals)):
        if "DOWN" in xVals[idx][cstIdx]:
            xVals[idx][cstIdx] = 0
        elif "FLAT" in xVals[idx][cstIdx]:
            xVals[idx][cstIdx] = 1
        elif "UP" in xVals[idx][cstIdx]:
            xVals[idx][cstIdx] = 2

        xVals[idx][ficoIdx] = int(xVals[idx][ficoIdx].split("-")[0])

        '''
        if "true" in xVals[idx][5]:
            xVals[idx][5] = 0
        elif "false" in xVals[idx][5]:
            xVals[idx][5] = 1
        ''' 
        xVals[idx].append(avgScores[xVals[idx][loanClassIdx]])
        
        #ch1 = xVals[idx][loanClassIdx][0]
        #xVals[idx][loanClassIdx] = (ord(ch1)-64) + (.2 * int(xVals[idx][loanClassIdx][1]))-.1

        #try:
        #    xVals[idx] = list(map(float, xVals[idx]))
        #except:
        #    print(xVals[idx])
        #    delIdx.append(idx)
            
    for idx in reversed(delIdx):
        del(xVals[idx])
        del(ids[idx])
        del(yVals[idx])

    print("Calculating Values...")
    for idx in range(len(xVals)):
        irrInputs = {'originalNote': float(xVals[idx][5]),
                     'csTrend': xVals[idx][1],
                     'FICO': xVals[idx][2],
                     'loanClass': xVals[idx][3],
                     'loanTerm': int(xVals[idx][4]),
                     'rate': float(xVals[idx][6]),
                     'remainingPmts': xVals[idx][7],
                     'askPrice': yVals[idx],
                     'avgScore': xVals[idx][8],
                     'principal': xVals[idx][0]}

        rate = calcIRR(irrInputs)
        
        #print("\nRate: {:.3}".format(rate))
        xVals[idx].append(rate)
        #print("TEST: ", xVals[idx])
        
    sortedList = sorted(((e,i) for i,e in enumerate(xVals)), key = lambda x: x[0][-1], reverse = True)

    results = {'data':[]}
    for row in sortedList[:5]:
        #print("\t",row[0])
        currentResult = { 'loanid':ids[row[1]][0],
                          'noteid':ids[row[1]][1],
                          'orderid':ids[row[1]][2],
                          'ask':yVals[row[1]],
                          'rate':row[0][-1],
                          'months':int(row[0][-3]),
                          'history':scrape.getPaymentHistory(ids[row[1]][0], ids[row[1]][2], ids[row[1]][1]) }

        print(currentResult)

        results['data'].append(currentResult)
        #print(ids[row[1]], "Ask : {}".format(yVals[row[1]]), "Rate: {:.3g}".format(row[0][-1]), "Months Remaining: {}".format(int(row[0][-3])))

        time.sleep(5)

    with open("currentResults.json", "w") as f:
        json.dump(results, f, indent="\t")
        
        
if __name__ == "__main__":

    available = apiTools.availableCash(config)
    #available = 20
    
    #reg = regression()
    findLoans(available)
