#/usr/bin/python3

import sqlite3
import re

from collections import defaultdict
from itertools import groupby
from statistics import mean, median, variance
from time import sleep

#from scipy.stats import ttest_ind, ks_2samp, levene, ansari

import matplotlib.pyplot as plt
import numpy, random

import apiTools, scrape

con = sqlite3.connect('current.db')
cur = con.cursor()

con2 = sqlite3.connect('history.db')
cur2 = con2.cursor()

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

def loadHistorical(sampleSize = 2500):
    sql = """insert or ignore into loans (LoanId, NoteId, OrderId,
                    OutstandingPrincipal, AccruedInterest, Status, AskPrice,
                    Markup, YTM, DaysSinceLastPayment, CreditScoreTrend,
                    FICO, Listed, NeverLate, LoanClass, LoanMaturity,
                    OriginalNoteAmount, InterestRate, RemainingPayments,
                    PrincipalInterest, ApplicationType, pmtHistoryScore) values
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    
    maxRow = cur.execute('select max(rowid) from loans').fetchone()[0]
    
    selected = list(map(str, random.sample(range(maxRow), sampleSize))) 
    
    cur.execute('select * from loans where rowid in ({}) and not Status = "Issued"'.format(','.join(selected)))
    results = cur.fetchall()
    
    newResults = []
    stepSize = int(sampleSize / 20)
    for cnt, row in enumerate(results):
        if cnt % stepSize == 0:
            print("Processing Sample", cnt+1)
    
        try:
            phScore = scrape.getPaymentHistory(*row[:3])
        except:
            print("\nPayment History Error Caught.  Continuing...")
            print("Loading {} Results".format(len(newResults)))
            cur2.executemany(sql, newResults)
            con2.commit()

            phScore = None
            newResults = []
            
        if phScore is None:
            print("Error: ", row[:3])
            #break
        
        sleep(5)
            
        newRow = list(row) + [phScore]
        newResults.append(newRow)
        
    print("Loading {} Results".format(len(newResults)))
    cur2.executemany(sql, newResults)
    con2.commit()
        
                
def processScores(inlist):
    cleanScores = [x[0].replace('"','') for x in inlist]
    splitScores = [int(x.split("-")[0]) for x in cleanScores]

    return splitScores

if __name__ == "__main__":

    apiTools.getCurrentNotes()
    loadActive("currentNotes.csv")

    #loadHistorical()
