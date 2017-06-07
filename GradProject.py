
import numpy as np
from itertools import combinations
from math import log
from math import exp
from operator import itemgetter
#import argparse
import heapq


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Print Welcome Message %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def welcomeMessage():
    print("Welcome to Doktar!")
    checkIfNumb=True
    print("What would be your choice as the question asking strategy for DD?")
    print("1- Entropy-Based")
    print("2- Disease-Based")
    print("3- Symptom-Based")
    choice=input("Please choose one (1/2/3): ")
    while checkIfNumb:
        try:
            choice =int(choice)
        except BaseException:
            print("Do not enter non-numeric characters!!!")
            choice=input("Please enter a valid choice (1/2/3): ")
        else:
            if int(choice)>0 and int(choice)<4:
                checkIfNumb=False
            else:
                print("Do not enter out-range numbers!!!")
                choice=input("Please enter a valid choice (1/2/3): ")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate All Possible Configurations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate all possible disease combinations, that is, given all the diseases and the maximum number of concurrent 
# diseases, every possible disease configuration is generated.
# 
# Output format is:
# 
# 

def generateAllCombs(Max, N):
    size = 0
    b = np.zeros((1,N), dtype=int)
    firstIndex = 0
    for x in range(0,Max+1):
        a = list(combinations(range(0,N),x))
        size += len(a)
        b.resize(size,N)
        for i in range(firstIndex,firstIndex+len(a)):
            for y in range(0,x):
                b[i][a[i-firstIndex][y]]=1

        firstIndex += len(a)
    return b, size

#usage:
# aaa, M = generateAllCombs(5,5)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate Log of Prior %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def evaluateLogOfPrior(diseaseList):
    size = len(diseaseList)
    tmp = size*log(priorProbOfADis) + (numOfDiseases-size)*log(1-priorProbOfADis)
    return tmp

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate Log of Likelihood %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def evaluateLogOfLikelihood(symptomList, diseaseList):
    LogLikelihood    = float(0)
    logOne           = float(0)
    numOfNotObserved = float(0)
    sumOfDisSymMatch = float(0)
    numOfDisSymMatch = float(0)
    for [sym, ans] in symptomList:
        numOfDisSymMatch = float(0)
        for dis in diseaseList:
            if sym in diseases2symptoms[dis]:
                numOfDisSymMatch+=1
        if ans==0:
            numOfNotObserved+=1
            sumOfDisSymMatch += numOfDisSymMatch
        else:
            logOne += log(1-probNotOfObservingJWhenNoDisPresent*pow(probNotOfObservingJWhenDisPresent,numOfDisSymMatch))
    LogLikelihood = numOfNotObserved*log(probNotOfObservingJWhenNoDisPresent) + sumOfDisSymMatch*log(probNotOfObservingJWhenDisPresent) + logOne
    return LogLikelihood

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute Evidence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def computeEvidence(symptomList):
    pE = float(0)
    #tmpDis = np.zeros((numOfAllDisCombs,numOfDiseases), dtype=int)
    tmpDis = []
    for i in range(0,numOfAllDisCombs):
        tmpDis = []
        for j in range(0,numOfDiseases):
            if allDiseaseCombs[i][j] == 1:
                tmpDis.append(j)
        pE += (exp(evaluateLogOfLikelihood(symptomList, tmpDis)))*(exp(evaluateLogOfPrior(tmpDis)))
    return pE

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute Marginal Table %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def computeMarginalTable(symptomList):
    margTable = np.zeros((2,numOfDiseases), dtype=float)

    logP = np.zeros(numOfAllDisCombs, dtype=float)
    normLogP  = np.zeros(numOfAllDisCombs, dtype=float)

    currDis = []
    for i in range(0,numOfAllDisCombs):
        currDis = []
        for j in range(0,numOfDiseases):
            if allDiseaseCombs[i][j] == 1:
                currDis.append(j)
        logP[i] = evaluateLogOfLikelihood(symptomList, currDis) + evaluateLogOfPrior(symptomList)

    #normalize
    logPMax = float(0)
    logPSum = float(0)
    normPSum = float(0)

    logPMax = max(logP)
    for i in range(0, numOfAllDisCombs):
        normLogP[i] = exp(logP[i]-logPMax)
    logPSum = sum(normLogP)
    normLogP = [x/logPSum for x in normLogP]

    for i in range(0,numOfAllDisCombs):
        for j in range(0, numOfDiseases):
            if allDiseaseCombs[i][j]==1:
                margTable[0][j] +=normLogP[i]
            else:
                margTable[1][j] +=normLogP[i]

    return margTable

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Question SYM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generateQuestionSym():
    questionToBeAsked = 0
    for sym, occurance in questionsListSymptom:
        if [sym,0] in symptoms:
            continue
        elif [sym,1] in symptoms:
            continue
        else:
            questionToBeAsked=sym
            break
    else:
        questionToBeAsked = -1
    return questionToBeAsked

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Question DIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generateQuestionDis():
    found = 0
    questionToBeAsked = 0
    for dis, occurance in questionsListDisease:
        for id in range(0,len(diseases2symptoms[dis])):
            sym=diseases2symptoms[dis][id]
            if [sym,0] in symptoms:
                continue
            elif [sym,1] in symptoms:
                continue
            else:
                questionToBeAsked=sym
                found=1
                break
        if found==1:
            break

    else:
        questionToBeAsked = -1


    return questionToBeAsked


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Question ENTSYM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generateQuestionEntSym():
    pEon = float(0)
    pEoff = float(0)
    Hon = float(0)
    Hoff = float(0)
    questionsAsInput = []
    enlisted = []
    numOfQuestionsToBeTested = np.floor(numOfSymptoms*0.25)
    for k in range(0,numOfQuestionsToBeTested):
        questionToBeAsked = 0
        for sym, occurance in questionsListSymptom:
            if [sym, 0] in symptoms:
                continue
            elif [sym, 1] in symptoms:
                continue
            elif sym in enlisted:
                continue
            else:
                questionToBeAsked = sym
                enlisted.append(sym)
                break
        else:
            questionToBeAsked = -1
        questionsAsInput.append(questionToBeAsked)


    questionsListEntropy = []
    PriorTable = computeMarginalTable(symptoms)
    for j in questionsAsInput:
        pEon = 0.0
        pEoff = 0.0
        Hon = 0.0
        Hoff = 0.0
        if [j, 0] in symptoms:
            continue
        elif [j, 1] in symptoms:
            continue
        elif j==-1:
            break
        else:
            pass
        symptoms.append([j, 1])
        PosteriorTableOn = computeMarginalTable(symptoms)
        for k in range(0, numOfDiseases):
            Hon += PosteriorTableOn[0][k] * (log(PosteriorTableOn[0][k]) - log(PriorTable[0][k]))
        pEon = computeEvidence(symptoms)
        symptoms.pop()
        symptoms.append([j, 0])
        PosteriorTableOff = computeMarginalTable(symptoms)
        for k in range(0, numOfDiseases):
            Hoff += PosteriorTableOff[0][k] * (log(PosteriorTableOff[0][k]) - log(PriorTable[0][k]))
        pEoff = computeEvidence(symptoms)
        questionsListEntropy.append([j, Hon * pEon + Hoff * pEoff])
        symptoms.pop()

    questionsListEntropy = sorted(questionsListEntropy, key=itemgetter(1), reverse=True)
    try:
        return questionsListEntropy[0][0]
    except BaseException:
        return -1


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Question ENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generateQuestionEnt():
    pEon  = float(0)
    pEoff = float(0)
    Hon   = float(0)
    Hoff  = float(0)
    
    questionsListEntropy = []
    PriorTable = computeMarginalTable(symptoms)
    for j in range(0, numOfSymptoms):
        pEon = 0.0
        pEoff = 0.0
        Hon = 0.0
        Hoff = 0.0
        if [j,0] in symptoms:
            continue
        elif [j,1] in symptoms:
            continue
        else:
            pass
        symptoms.append([j,1])
        PosteriorTableOn = computeMarginalTable(symptoms)
        for k in range(0, numOfDiseases):
            Hon += PosteriorTableOn[0][k]* (log(PosteriorTableOn[0][k])-log(PriorTable[0][k]))
        pEon = computeEvidence(symptoms)
        symptoms.pop()
        symptoms.append([j,0])
        PosteriorTableOff = computeMarginalTable(symptoms)
        for k in range(0, numOfDiseases):
            Hoff += PosteriorTableOff[0][k]* (log(PosteriorTableOff[0][k])-log(PriorTable[0][k]))
        pEoff = computeEvidence(symptoms)
        questionsListEntropy.append([j,Hon*pEon+Hoff*pEoff])
        symptoms.pop()


    questionsListEntropy = sorted(questionsListEntropy, key=itemgetter(1), reverse= True)
    try:
        return questionsListEntropy[0][0]
    except BaseException:
        return -1
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Question  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generateQuestion(choice):
    if choice==1:
        return generateQuestionEnt()
    elif choice==2:
        return generateQuestionDis()
    elif choice==3:
        return generateQuestionSym()
    elif choice==4:
        return generateQuestionEntSym()
    else:
        return -1



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Infer Best K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def inferBestK():
    global diseases
    global bestDiag
    global logPForFinal
    bestDiag = []

    logPForFinal = float(0)
    diseases = []
    logPForFinal = evaluateLogOfLikelihood(symptoms, diseases) + evaluateLogOfPrior(diseases)
    storeHeap()
    if maxNumOfConcDisesases in range(1,4):
        for i in range(1,numOfDiseases+1):
            diseases = []
            diseases.append(i-1)
            logPForFinal = evaluateLogOfLikelihood(symptoms, diseases) + evaluateLogOfPrior(diseases)
            storeHeap()
            if maxNumOfConcDisesases in range(2,4):
                for j in range(1,i):
                    diseases = []
                    diseases.append(j-1)
                    diseases.append(i-1)
                    logPForFinal = evaluateLogOfLikelihood(symptoms, diseases) + evaluateLogOfPrior(diseases)
                    storeHeap()
                    if maxNumOfConcDisesases in range(3,4):
                        for k in range(1,j):
                            diseases = []
                            diseases.append(k-1)
                            diseases.append(j-1)
                            diseases.append(i-1)
                            logPForFinal = evaluateLogOfLikelihood(symptoms, diseases) + evaluateLogOfPrior(diseases)
                            storeHeap()
    bestDiag.sort()
    heapq._heapify_max(bestDiag) 


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Store Heap %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def storeHeap():
    global bestDiag

    if len(bestDiag)<numOfBestKConfs:
        bestDiag.append([logPForFinal,diseases])
        heapq.heapify(bestDiag)
    else:
        
        if bestDiag[0][0]<logPForFinal:
            heapq.heappop(bestDiag)
            heapq.heappush(bestDiag,[logPForFinal, diseases])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normalize Exp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalizExpression():
    global normalizedLogP
    global bestDiag
    global tmpBestList
    logPMax = float(0)
    logPSum = float(0)
    logPMax = bestDiag[0][0]
    normalizedLogP = []
    tmpBestList = []
    for i in range(0,numOfBestKConfs):
        normalizedLogP.append(np.exp(bestDiag[i][0]-logPMax))
        tmpBestList.append(bestDiag[i][1])
    logPSum = sum(normalizedLogP)
    normalizedLogP = [x/logPSum for x in normalizedLogP]
  #  print(normalizedLogP)
    return tmpBestList
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Print Diagnosis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def printDiagnosis():
    
    counter = 1
    for diags in bestDiag:
        
        
        if(len(diags[1])==0):
            print("#{} likely configuration is: NoDisease".format(counter))
        else:
            print("#{} likely configuration is: ".format(counter), end="")
            for dis in diags[1]:
                print(diseaseNames[dis], end="")
                print(" ",end="")
        counter+=1
    print("")





def getResultToTest(tmpSymptoms):
    global symptoms
    symptoms = []
    bestList = []
    for i in range(0,numOfSymptoms):
        symptoms.append([i,tmpSymptoms[i]])
    inferBestK()
    bestList = normalizExpression()
    symptoms = []
    return bestList

def Test(NumberOfTests):
    global symptoms
    global bestList
    global numOfQuestionsToConvergeToTheFinal
    global bestDiag
    numOfQuestionsToConvergeToTheFinal = np.zeros([4,NumberOfTests], dtype=int)
    for j in range(0,NumberOfTests):
        bestList = []
        #tmpSymptoms = np.random.randint(2, size=(numOfSymptoms))
        tmpSymptoms = np.random.choice(2, numOfSymptoms, p=[0.85, 0.15])
        bestList = getResultToTest(tmpSymptoms)
        diseases = []
        symptoms = []

        bestDiag = []
        logPForFinal = float(0)
        normalizedLogP = []
        print("++++++++J is {}".format(j))
        for i in range(1,5):
            print("I is {}".format(i))
            counter = 0
            diseases = []
            symptoms = []

            bestDiag = []
            logPForFinal = float(0)
            normalizedLogP = []
            inferBestK()
            tmpBestList = normalizExpression()

            
            choice=i

            while not bestList==tmpBestList:


                resp= generateQuestion(choice)
                if tmpSymptoms[resp]==1:
                    symptoms.append([resp,1])
                elif tmpSymptoms[resp]==0:
                    symptoms.append([resp,0])

                counter+=1
                inferBestK()
                tmpBestList = normalizExpression()
            numOfQuestionsToConvergeToTheFinal[i-1][j]=counter

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#parser = argparse.ArgumentParser()
#parser.add_argument("file", help="input file for network model", type=str)
#parser.add_argument("choice", help="mode of question asking strategy", type=int)
#parser.add_argument("inputMethod", help="whether the user provides the input or it's read from a file", type=int)

#args = parser.parse_args()
#fileName =  args.file
#choice = args.choice
#provider = args.inputMethod

fileName = "input.mesin"
choice = 1
provider = 1
if provider==1:
    run = True
else:
    run = False

#arr = np.random.randint(2, size=(20,10))


probNotOfObservingJWhenDisPresent   = 0.02 #th
probNotOfObservingJWhenNoDisPresent = 0.95 #th0
priorProbOfADis                     = 1 #prior
numOfBestKConfs                     = 5 
maxNumOfConcDisesases               = 3

diseaseNames      = []
symptomNames      = []

questionsListSymptom = []
questionsListDisease = []
questionsListEntropy = []

diseases = []
symptoms = []



tmpBestList = []
numOfQuestionsToConvergeToTheFinal = []
bestList = []
bestDiag = []
logPForFinal = float(0)
normalizedLogP = []

file = open(fileName, 'r')
numOfDiseases = int(file.readline())
numOfSymptoms = int(file.readline())


priorProbOfADis = 1/numOfDiseases

diseases2symptoms = {x: [] for x in range(0,numOfDiseases)}
symptoms2diseases = {x: [] for x in range(0,numOfSymptoms)}
for x in range(0,numOfDiseases):
    diseaseNames.append(str(file.readline()))
for x in range(0,numOfSymptoms):
    symptomNames.append(str(file.readline()))
for x in range(0,numOfSymptoms):
    line = file.readline()
    for y in range(0,2*numOfDiseases,2):
        tmp=int(line[y])
        print(tmp, end=" ")
        if tmp==1:
            symptoms2diseases[x].append(int(y/2))
            diseases2symptoms[int(y/2)].append(x)
    print("")
questionsListSymptom = sorted([(k, len(symptoms2diseases[k])) for k in sorted(symptoms2diseases, key=symptoms2diseases.get)], key=lambda x: x[1], reverse=True)
questionsListDisease = sorted([(k, len(diseases2symptoms[k])) for k in sorted(diseases2symptoms, key=diseases2symptoms.get)], key=lambda x: x[1], reverse=True)
questionsListEntropy = []
allDiseaseCombs, numOfAllDisCombs = generateAllCombs(maxNumOfConcDisesases, numOfDiseases)
file.closed



def welcomeMessage():
    print("Welcome to Doktar!")
    checkIfNumb=True
    print("What would be your choice as the question asking strategy for DD?")
    print("1- Entropy-Based")
    print("2- Disease-Based")
    print("3- Symptom-Based")
    choice=input("Please choose one (1/2/3): ")
    while checkIfNumb:
        try:
            choice =int(choice)
        except BaseException:
            print("Do not enter non-numeric characters!!!")
            choice=input("Please enter a valid choice (1/2/3): ")
        else:
            if int(choice)>0 and int(choice)<4:
                checkIfNumb=False
            else:
                print("Do not enter out-range numbers!!!")
                choice=input("Please enter a valid choice (1/2/3): ")
diss = 2

#for i in range(0,numOfSymptoms):
    #symptoms.append([i,0])


avgNumOfQuestionsToConvergeToTheFinal  =  np.zeros([1, 4], dtype=float)
for i in range(0,1):
    print("Sim {} is running".format(i))
    Test(30)
    avgNumOfQuestionsToConvergeToTheFinal[i,:] = np.mean(numOfQuestionsToConvergeToTheFinal, axis=1)

print(avgNumOfQuestionsToConvergeToTheFinal)




#for i in range(0,len(diseases2symptoms[diss])):
 #   symptoms[diseases2symptoms[diss][i]][1] = 1
#    inferBestK()
#    normalizExpression()
#    print("TMP{} LIST".format(i+1))
#    print(tmpBestList)
#    print("STEP{}".format(i+1))
#    printDiagnosis()


