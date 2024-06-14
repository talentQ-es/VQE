import json

def qspain():
    
    with open('./styles/quantum-spain.json') as json_file:
        qspain = json_file.read()
        
    qspain = json.loads(qspain)

    return qspain