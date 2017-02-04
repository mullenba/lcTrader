import requests
import session
import re
import statistics 
import pickle

from bs4 import BeautifulSoup


with open('sessionAuth.pkl', 'rb') as f:
    authStrings = pickle.load(f)

curSession = session.Session()
curSession.authenticate(*authStrings)

def getPaymentHistory(loanid, orderid, noteid, savePage = False):

    valueDict = {   'Completed' : 5.0,
                    'Completed - In Grace Period' : 3.0,
                    'Completed - Late (16-30 days)' : 2.0,
                    'Completed - Late (31-120 days)': 1.0,
                    'Not Received' : 0.0,
                    'Partial Payment - Late (31-120 days)' : 0.5}

    params = {'showfoliofn':'true', 'loan_id':loanid, 'order_id':noteid, 'note_id':orderid}
    r = curSession.request('POST','foliofn/browseNotesLoanPerf.action', query=params)
    
    if savePage:
        with open("debug.txt", "w") as f:
            f.write(r.text)
    
    soup = BeautifulSoup(r.text, "html.parser")

    paymentHistory = soup.find_all(id="lcLoanPerfTable1")
    
    scores = []
    
    if paymentHistory == []: return None
    
    for row in paymentHistory[0].find_all('tr'):
        rowText = []
        for cell in row.find_all('td'):
            rowText.append(cell.string)

        if len(rowText) == 0: continue  # Nothing useful on the line

        if rowText[-2].strip() == "Scheduled": continue  # Don't apply this to the score.
            
        try:
            #print(rowText[-2].strip())
            scores.append(valueDict[rowText[-2].strip()]) 
        except:
            pass
    
    if len(scores) == 0: return None
    finalScore = statistics.mean(scores)
    #print("Final Score: ", finalScore)
    
    return finalScore

def testFromFile(filename):
    valueDict = {   'Completed' : 5.0,
                    'Completed - In Grace Period' : 3.0,
                    'Completed - Late (16-30 days)' : 2.0,
                    'Completed - Late (31-120 days)': 1.0,
                    'Not Received' : 0.0,
                    'Partial Payment - Late (31-120 days)' : 0.5}
    
    with open(filename, "r") as f:
        fileText = f.read()

    soup = BeautifulSoup(fileText)

    paymentHistory = soup.find_all(id="lcLoanPerfTable1")
    scores = []
    
    if paymentHistory == []: return None
    
    for row in paymentHistory[0].find_all('tr'):
        rowText = []
        for cell in row.find_all('td'):
            rowText.append(cell.string)

        if len(rowText) == 0: continue

        if rowText[-2].strip() == "Scheduled": continue
        
        print(valueDict[rowText[-2].strip()])
            
        try:
            #print(rowText[-2].strip())
            scores.append(valueDict[rowText[-2].strip()]) 
        except:
            pass
    
    if len(scores) == 0: return None
    finalScore = statistics.mean(scores)
    #print("Final Score: ", finalScore)
    
    return finalScore
        
if __name__ == "__main__":
    #(7955128, 32984932, 13002276)
    #(42303298, 73387695, 58192212)
    #(42394992, 73872012, 58653233)
    #(49188567, 81690314, 108565042)

    print(getPaymentHistory(60095699, 94968643, 83410761, True))

    #print(testFromFile("debug.txt"))
