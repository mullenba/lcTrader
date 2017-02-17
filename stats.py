import numpy as np
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt

con = sqlite3.connect('current.db')
cur = con.cursor()

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
        tableVals.append((payment, pv))
        
    return tableVals

def PV(fv, rate, years):
    pVal = fv / np.power((1 + rate), years)
    return pVal

def NPV(initial_investment, principal, rate):
    pVal = -1.0 * initial_investment

    npVal = pv((principal * rate), rate, 1)

    print(npVal, pVal)

    return (npVal - pVal)

def IRR(initial_investment, principal, note_val, rate, periods, oTerm):
    pv = -1.0 * initial_investment

    npv = 1
    while abs(npv) > 0.001:
        pmt = calcPayment(note_val, rate * 1200, oTerm)
        aTbl = amortTable(principal, pmt, rate * 1200, periods)
        npv = pv + sum([x[0] for x in aTbl])
        rate -= npv / 1000
        #print(npv)

    return rate

def survival():
    fields = [  'LoanId', 'NoteId', 'OrderId','OutstandingPrincipal', 'AskPrice',
                'YTM', 'CreditScoreTrend','FICO', 'LoanClass','OriginalNoteAmount', 'InterestRate', 'RemainingPayments',
                'PrincipalInterest', 'ApplicationType', 'Status']

    cur.execute('select {} from loans where not status like "%Issued%" and not (status = "Current" and NeverLate = 0) and LoanMaturity = 36 and RemainingPayments < 36'.format(",".join(fields)))

    results = []
    for row in cur.fetchall():
        if row[-1] == "Current":
            censored = 0
        else:
            censored = 1
        results.append([(row[0], row[1], row[2]), 36-int(row[11]), censored, row[8]])

    results.sort(key=lambda x: x[1])

    grades = []

    dataset = defaultdict(lambda: defaultdict(int))
    for row in results:
        dataset[row[1]][row[2]] += 1
        grades.append(row[3])

    #grades = sorted(list(set(grades)))
    #print(grades)

    probs = []
    for row in dataset.items():
        nc = row[1][0]
        c = row[1][1]

        for idx in range(row[0]+1,35):
            nc += dataset[idx][0]
            nc += dataset[idx][1]

        probs.append(float(nc) / float(nc + c))

    for row, cumprod in zip(dataset.items(), np.cumprod(probs)):
        print(row[0], cumprod)

    
if __name__ == "__main__":
    interest = 8.9

    #print(PV(900, (.10/12), 36))
    #print(NPV(4.91, 4.93, (.1199 / 12)))

    #irr = IRR(21.19, 23.24, 25, (.2099 / 12), 53, 60)
    #print("APY = {:.2}%".format(irr * 1200))
    #print(IRR(4.91, 4.93, 50, (0.10 / 12), 3))

    survival()
    
