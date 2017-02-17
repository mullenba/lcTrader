import requests, pickle

import json



def getCurrentNotes():
    url = 'https://resources.lendingclub.com/SecondaryMarketAllNotes.csv'

    with open("authkeys.pkl", "rb") as f:
        headers = pickle.load(f)


    print("Sending request: ", url)
    r = requests.get(url, headers=headers, stream=True)

    with open("currentNotes.csv", 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

    return "currentNotes.csv"

def availableCash(config):
    url = "https://api.lendingclub.com/api/investor/v1/accounts/{}/availablecash"
    
    headers = {'Authorization': config.get('account_info', 'api_key')}
    accountNumber = config.get('account_info', 'account')

    r = requests.get(url.format(accountNumber), headers=headers)

    try:
        rObj = r.json()
    except:
        print(r.text)

    return rObj['availableCash']

def orderNotes(noteID, price, acctID):
    url = "https://api.lendingclub.com/api/investor/v1/accounts/{}/trades/buy"
    
    with open("authkeys.pkl", "rb") as f:
        headers = pickle.load(f)

    order = {'aid':acctID,
               'notes':[ {'loanId':noteID[0], 'orderId':noteID[2], 'noteId':noteID[1], 'bidPrice':price} ]
               }

    payload = json.dumps(order, indent='\t')

    r = requests.post(url.format(acctID), headers=headers, data=payload)

    print(r.text)

def saveHeaders(authKey):
    headers = {'Content-Type': 'application/json', 'Authorization':authKey}

    with open("authkeys.pkl", "wb") as f:
        pickle.dump(headers, f)


    return



if __name__ == "__main__":
    #saveHeaders('XXXXXXXXXXXXXXX')

    orderNotes([49188158, 81657352, 67852036], 0.99, 101133232)

    test = {"buyNoteConfirmations":[{"loanId":49188158,
                                     "noteId":81657352,
                                     "bidPrice":0.99,
                                     "outstandingAccruedInterest":0,
                                     "outstandingPrincipal":0.99,
                                     "yeildToMaturity":0,
                                     "executionStatus":["SUCCESS_PENDING_SETTLEMENT"],
                                     "txnId":"b355f3c5c-48f3-4b3c-8bb1-ee9de0aae5b4"}]}
    
    pass
