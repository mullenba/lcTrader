import sqlite3
import re, pickle

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

con = sqlite3.connect('current.db')
cur = con.cursor()

hcon = sqlite3.connect('history.db')
hcur = hcon.cursor()


import warnings
warnings.filterwarnings("ignore")


def loadActive(filename):
    try:
        cur.execute("delete from loans")
    except:
        pass
    
    cur.execute("vacuum")


    sql = """create table if not exists loans (LoanId INT, NoteId INT, OrderId INT,
                    OutstandingPrincipal REAL, AccruedInterest REAL, Status TEXT, AskPrice REAL,
                    Markup REAL, YTM REAL, DaysSinceLastPayment INT, CreditScoreTrend TEXT,
                    FICO TEXT, Listed TEXT, NeverLate INT, LoanClass TEXT, LoanMaturity TEXT,
                    OriginalNoteAmount TEXT, InterestRate REAL, RemainingPayments INTEGER,
                    PrincipalInterest REAL, ApplicationType TEXT)"""
    
    cur.execute(sql)

    sql = """insert into loans (LoanId, NoteId, OrderId,
                    OutstandingPrincipal, AccruedInterest, Status, AskPrice,
                    Markup, YTM, DaysSinceLastPayment, CreditScoreTrend,
                    FICO, Listed, NeverLate, LoanClass, LoanMaturity,
                    OriginalNoteAmount, InterestRate, RemainingPayments,
                    PrincipalInterest, ApplicationType) values
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    with open(filename, "r") as f:
        header = True
        rowData = []
        for idx, line in enumerate(f):
            if header:
                header = False
                continue
            rowData.append(line.strip().replace('"','').split(','))

            if idx > 0 and idx % 1000 == 0:
                cur.executemany(sql, rowData)
                con.commit()
                rowData = []
                print(idx)

def processScores(inlist):
    cleanScores = [x[0].replace('"','') for x in inlist]
    splitScores = [int(x.split("-")[0]) for x in cleanScores]

    return splitScores

def fico_default():
    #goodStatus = ['Fully Paid','Current','In Grace Period','Does not meet the credit policy. Status:Fully Paid']
    #badStatus = ['Charged Off','Default','Late (31-120 days)','Late (16-30 days)','Does not meet the credit policy. Status:Charged Off']
    #neutralStatus = [None, 'Issued']

    cur.execute('select FICO, status from loans where not status like "%Issued%"')

    uKeys = []
    groups = []
    data = sorted(cur.fetchall(), key=lambda x: x[1])
    for k, g in groupby(data, lambda x: x[1]):
        scores = list(g)
        groups.append(processScores(scores))      # Store group iterator as a list
        uKeys.append(k)

    '''
    for i in range(len(groups)):
        transformData = list(map(log, groups[i]))
        print(uKeys[i], mean(transformData), variance(transformData))
    '''

    for i in range(1, len(groups)):
        print(uKeys[0], " vs ", uKeys[i])
        #transformData1 = list(map(log, groups[i-1]))
        #transformData2 = list(map(log, groups[i]))
        print(ttest_ind(groups[0], groups[i], False))

    '''
    plt.hist(list(map(log, groups[0])), bins=30, histtype='stepfilled', normed=True, color='b', label=uKeys[0])
    plt.hist(list(map(log, groups[3])), bins=30, histtype='stepfilled', normed=True, color='r', alpha=0.5, label=uKeys[3])
    #plt.title("Gaussian/Uniform Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    '''

def priceRegression():
    fields = [  'LoanId', 'NoteId', 'OrderId','OutstandingPrincipal', 'CreditScoreTrend','FICO',
                'LoanClass', 'LoanMaturity','OriginalNoteAmount', 'InterestRate', 'RemainingPayments', 'Status', 'AskPrice']

    print("Loading Data...")
    cur.execute('select {} from loans'.format(",".join(fields)))

    ids = []
    xVals = []
    yVals = []

    for row in cur.fetchall():
        ids.append(row[:3])
        xVals.append(list([x for x in row[3:-1]]))
        
        
        yVals.append(float(row[-1]))
        
    print("Cleaning Variables...")
    # Clean up variables
    
    delIdx = []
    
    cstIdx = 1
    loanClassIdx = 3
    ficoIdx = 2
    statusIdx = 8
    
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
            
        ch1 = xVals[idx][loanClassIdx][0]
        xVals[idx][loanClassIdx] = (ord(ch1)-64) + (.2 * int(xVals[idx][loanClassIdx][1]))-.1

        if xVals[idx][statusIdx] == "Issued":
            xVals[idx][statusIdx] = 0
        elif xVals[idx][statusIdx] == "Current":
            xVals[idx][statusIdx] = 1
        elif xVals[idx][statusIdx] == "In Grace Period":
            xVals[idx][statusIdx] = 2
        elif xVals[idx][statusIdx] == "Late (16-30 days)":
            xVals[idx][statusIdx] = 3
        elif xVals[idx][statusIdx] == "Late (31-120 days)":
            xVals[idx][statusIdx] = 4
        
        try:
            xVals[idx] = list(map(float, xVals[idx]))
        except:
            print(xVals[idx])
            delIdx.append(idx)
            
    for idx in reversed(delIdx):
        del(xVals[idx])
        del(yVals[idx])

    print("Performing Regression...")
    cData = numpy.array([numpy.array(xi) for xi in xVals])

    reg = RandomForestRegressor(n_jobs=-1)
    reg.fit(cData, yVals)
    
    '''
    targets = []
    cur.execute("select LoanId, NoteId, OrderId from loans where Markup < 0")
    for row in cur.fetchall():
        targets.append((row[0],row[1],row[2]))
    targets = set(targets)
    
    print("Preparing Output...")
    transformedData = reg.predict(cData)
    results = {}
    for x,y,z in zip(ids, transformedData, yVals):
        if x in targets:
            results[x] = (y/z, y)

    for row in sorted(results.items(), key = lambda x: x[1][0])[:10]:
        print(row)
    '''
    
    return reg
   
def regression():

    fields = [  'LoanId', 'NoteId', 'OrderId','OutstandingPrincipal', 'CreditScoreTrend','FICO',
                'LoanClass', 'LoanMaturity','OriginalNoteAmount', 'InterestRate', 'RemainingPayments', 'pmtHistoryScore', 'Status']

    print("Loading Data...")
    hcur.execute('select {} from loans where pmtHistoryScore > 0'.format(",".join(fields)))

    ids = []
    xVals = []
    yVals = []

    for row in hcur.fetchall():
        ids.append(row[:3])
        xVals.append(list([x for x in row[3:-1]]))
        
        if row[-1] == "Current":
            censored = 0
        else:
            censored = 1
        yVals.append(censored)
        
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
            
        ch1 = xVals[idx][loanClassIdx][0]
        xVals[idx][loanClassIdx] = (ord(ch1)-64) + (.2 * int(xVals[idx][loanClassIdx][1]))-.1

        try:
            xVals[idx] = list(map(float, xVals[idx]))
        except:
            print(xVals[idx])
            delIdx.append(idx)
            
    for idx in reversed(delIdx):
        del(xVals[idx])
        del(yVals[idx])
            
    print("Performing Regression...")
    cData = numpy.array([numpy.array(xi) for xi in xVals])

    reg = LogisticRegression(solver='sag', n_jobs=-1, max_iter=200)
    reg.fit(cData, yVals)
    
    print("Score: ", reg.score(cData, yVals))
    print("Coeffs: ", reg.coef_)
    
    return reg
    
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
        
        tableVals.append((owedInterest, pv))
        
    return tableVals

def calcValue(loanVals, reg):
    amount, csTrend, fico, loanClass, periods, interest, remaining, score = loanVals
    
    monthlyPayment = calcPayment(amount, interest, periods)
    
    aTable = amortTable(amount, monthlyPayment, interest, periods)
    
    value = 1.0
    sumPayments = 0
    for idx, row in enumerate(aTable[-int(remaining):]):
        currentValues = [row[1], csTrend, fico, loanClass, periods, amount, interest, periods-1-idx, score]
        prediction = reg.predict_proba(currentValues)[0][0]
        value *= prediction
        sumPayments += row[0]
        #print(idx, currentValues, prediction)

    #print("Value: ", value)
    return value * sumPayments
     
def findLoans(reg, price):
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
        
        ch1 = xVals[idx][loanClassIdx][0]
        xVals[idx][loanClassIdx] = (ord(ch1)-64) + (.2 * int(xVals[idx][loanClassIdx][1]))-.1

        try:
            xVals[idx] = list(map(float, xVals[idx]))
        except:
            print(xVals[idx])
            delIdx.append(idx)
            
    for idx in reversed(delIdx):
        del(xVals[idx])
        del(ids[idx])
        del(yVals[idx])

    print("Calculating Values...")
    for idx in range(len(xVals)):
        regVars = [xVals[idx][5], xVals[idx][1], xVals[idx][2], xVals[idx][3], xVals[idx][4], xVals[idx][6], xVals[idx][7], xVals[idx][8]]
        value = calcValue(regVars, reg)
        rate = (numpy.power((value / yVals[idx]), (1.0 / xVals[idx][7])) -1)
        #print(rate, value, yVals[idx], xVals[idx][7])

        xVals[idx].append(rate)
        
    sortedList = sorted(((e,i) for i,e in enumerate(xVals)), key = lambda x: x[0][-1], reverse = True)
        
    for row in sortedList[:10]:
        #print("\t",row[0])
        print(ids[row[1]], "Ask : {}".format(yVals[row[1]]), "Rate: {:.3g}".format(row[0][-1]), "Months Remaining: {}".format(int(row[0][-3])))
        
        
        
if __name__ == "__main__":
    #apiTest.getCurrentNotes()
    #loadActive("currentNotes.csv")

    with open('accountNum.pkl', 'rb') as f:
        accountNumber = pickle.load(f)

    #available = apiTools.availableCash(accountNumber)
    available = 30
    
    reg = regression()
    findLoans(reg, available)
