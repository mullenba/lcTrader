import requests, pickle





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

def availableCash(accountNumber):
    url = "https://api.lendingclub.com/api/investor/v1/accounts/{}/availablecash"
    
    with open("authkeys.pkl", "rb") as f:
        headers = pickle.load(f)

    r = requests.get(url.format(accountNumber), headers=headers)

    rObj = r.json()

    return rObj['availableCash']

def saveHeaders(authKey):
    headers = {'Content-Type': 'application/json', 'Authorization':authKey}

    with open("authkeys.pkl", "wb") as f:
        pickle.dump(headers, f)


    return



if __name__ == "__main__":
    #saveHeaders('XXXXXXXXXXXXXXX')
    pass
