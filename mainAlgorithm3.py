#from pyspark import SparkContext
import math, sys, os, shutil, argparse, re, random

#------------------------- spark emulation procedures -----------

def findNextPositionKeyNonEqual(lst, index):
    while index + 1 < len(lst) and lst[index][0] == lst[index + 1][0]:
        index += 1
    return index + 1

class LocalSparkTable:
    def __init__(self, table):
        self.table = table

    def map(self, f):
        newTable = []
        for record in self.table:
            newTable.append(f(record))
        return LocalSparkTable(newTable)

    def flatMap(self, f):
        newTable = []
        for record in self.table:
            lst = f(record)
            for recordLst in lst:
                newTable.append(recordLst)
        return LocalSparkTable(newTable)

    def join(self, table):
        lst1 = self.table
        lst1.sort()
        lst2 = table.table
        lst2.sort()

        index1 = 0
        index2 = 0
        answer = []
        while (index1 < len(lst1) and index2 < len(lst2)):
        #    print index1, ' ', index2, ' ', lst1[index1][0], ' ', lst2[index2][0], ' ', type(lst1[index1][0]), ' ', type(lst2[index2][0]) 
            if lst1[index1][0] < lst2[index2][0]:
                #print 'WAH 1'
                index1 += 1
            elif lst1[index1][0] > lst2[index2][0]:
                #print 'WAH 2'
                index2 += 1
            else:
                #print 'OK ', lst1[index1][0]
                nextIndex1 = findNextPositionKeyNonEqual(lst1, index1)
                nextIndex2 = findNextPositionKeyNonEqual(lst2, index2)
                while index1 < nextIndex1:
                    for i in range(index2, nextIndex2):
                        answer.append((lst1[index1][0], (lst1[index1][1], lst2[i][1])))
                    index1 += 1
                index2 = nextIndex2
        return LocalSparkTable(answer)

    def takeSample(self, withReplacement, number, seed = 57179):
        random.seed(seed)
        random.shuffle(self.table)
        ans = self.table[0:min(number, len(self.table))]
        if withReplacement:
            self.table = ans
        return ans

    def count(self):
        return len(self.table)

    def reduceByKey(self, f):
        #print 'reduceByKey started'
        index = 0
        answer = []
        self.table.sort()
        while index < len(self.table):
            nextIndex = findNextPositionKeyNonEqual(self.table, index)
            record = self.table[index]
            index += 1
            while index < nextIndex:
                record = record[0], f(record[1], self.table[index][1])
                index += 1
            answer.append(record)
        #print 'reduceByKey finished'
        return LocalSparkTable(answer)

    def reduce(self, f):
        #print 'reduce started'
        record = self.table[0]
        for index in range(1, len(self.table)):
            record = f(record, self.table[index])
        #print 'reduce result: ',record
        return record

    def collect(self):
        return self.table

class LocalSparkContext:
    def textFile(self, fileName):
        ans = []
        f = open(fileName, 'r')
        for line in f:
            ans.append(line[:-2])
        return LocalSparkTable(ans)

    def parallelize(self, lst):
        return LocalSparkTable(lst)

#----------------------------------------------------------------


def calcHash(string, hash_modulo, hash_osn):
    ans = 0
    for ch in string:
        ans = (ans * hash_osn + ord(ch)) % hash_modulo
    return ans

def calcID(strValue):
    return calcHash(strValue, 10**15 + 7, 179)

def getValueOr0(value):
    if value is None:
        return 0.0
    return value

def aggregateSum(value1, value2):
    return getValueOr0(value1) + getValueOr0(value2)

def aggregateSumValues(record1, record2):
    return record1[0], getValueOr0(record1[1]) + getValueOr0(record2[1])

def sign(x):
    if x == 0.0:
        return 0
    if x < 0.0:
        return -1
    return 1

def makeIDClickOrFeaturesValues(parameters, isIDHash, strValue, isIDClick):
    columnsNumber = 1 + parameters.catFeatures + parameters.numFeatures
    fields = strValue.split(parameters.delimiter)
    if len(fields) != columnsNumber or not fields[0].isdigit():
        return []
    
    if isIDHash:
        ID = calcID(strValue)
    else:
        ID = fields[0]
    
    if isIDClick:
        return [(ID, int(fields[0]) * 2 - 1)]
    else:
        answer = [(0, (ID, 1.0))]
        for i in range(1, parameters.catFeatures + 1):
            if fields[i] != '':
                featureName = 1 + parameters.numFeatures + parameters.catFeatureVariants * (i - 1) + \
                                  calcHash(fields[i], parameters.catFeatureVariants, 179)
                answer.append((featureName, (ID, 1.0)))
        for i in range(parameters.catFeatures + 1, columnsNumber):
            if fields[i] != '':
                try:
                    f = float(fields[i])
                except ValueError:
                    continue
                answer.append((i - parameters.catFeatures, (ID, float(fields[i]))))
        return answer
               
# -------------- punkt 0 ------------------------
def makeIDClick(parameters, isIDHash):
    def f(strValue):
        return makeIDClickOrFeaturesValues(parameters, isIDHash, strValue, True)
    return f

def makeIDClickRDD(trainInputRDD, parameters, isIDHash = True):
    #print 'LBFGS: maeIDClickRDD started'
    #print trainInputRDD.count()
    #print trainInputRDD.take(10)

    f = makeIDClick(parameters, isIDHash)
    trainClickRDD = trainInputRDD.flatMap(f)
    #print 'LBFGS: maeIDClickRDD found'
    #print trainInputRDD.count()
    #lst = trainInputRDD.take(10)
    #print lst
    #for x in lst:
    #    x = f(x)
    #print lst
    #print 'DONE makeIDClickRDD'
    return trainClickRDD
     
# ------------- end of punkt 0 -----------------

# ------------- punkt 1 ------------------------
def makeFeaturesValues(parameters, isIDHash):
    def f(strValue):
        return makeIDClickOrFeaturesValues(parameters, isIDHash, strValue, False)
    return f

def calcMax(p1, p2):
    value1 = math.fabs(p1[1])
    value2 = math.fabs(p2[1])
    if (value1 > value2):
        return p1
    else:
        return p2

def makeFeaturesValuesRDD(trainInputRDD, parameters, isIDHash):
    trainRDD = trainInputRDD.flatMap(makeFeaturesValues(parameters, isIDHash))
    return trainRDD
# ------------- end of punkt 1 -----------------
# ------------- punkt 2 ------------------------
def generateWCoefsInit(sc, trainRDD, parameters):
    wCoefsRDD = trainRDD.map(lambda x: (x[0], 0.0)).reduceByKey(lambda x, y: x)
    return wCoefsRDD
    #wCoefsRDD = wCoefsRDD.join(inputRDD)
    #print 'RAZ: ', wCoefsRDD.count()
    #print 'RAZ: ', wCoefsRDD.takeSample(False, 10, 35)
    
    #wCoefsRDD = wCoefsRDD.map(lambda x: (x[0], 0.0))
    #print 'DVA: ', wCoefsRDD.takeSample(False, 10, 35)
    #wCoefsRDD = wCoefsRDD.reduceByKey(lambda x, y: x)
    #print 'TRI: ', wCoefsRDD.takeSample(False, 10, 35)
    #return wCoefsRDD

def generateWCoefsStochasticGradient(sc, parameters):
    wCoefsList = [0.0] * (1 + parameters.catFeatureVariants * parameters.catFeatures + parameters.numFeatures)
    f = open(('/hdfs/' if parameters.hdfs else '') + parameters.inputTrainTable, 'r')
    nu = 0.1
    for line in f:
        line = line[0:len(line)-1]
        featuresList = makeIDClickOrFeaturesValues(parameters, False, line, False)
        #print 'featuresList: ', featuresList
        if not featuresList:
            continue
        y = float(featuresList[0][1][0]) * 2 - 1.0
        ys = 0.0
        for feature in featuresList:
            ys += wCoefsList[feature[0]] * feature[1][1]
        mult = 0.0
        if y * ys > 20:
            mult = y * math.exp(- y*ys)
        else:
            mult = y / (1.0 + math.exp(y * ys))
        for feature in featuresList:
            wCoefsList[feature[0]] += nu * mult * feature[1][1]


    f.close()
    len2 = 0.0
    for i in range(0, len(wCoefsList)):
        len2 = len2 + (wCoefsList[i] ** 2)
    
    len2 = len2 ** 0.5
    if len2 != 0.0:
        for i in range(0, len(wCoefsList)):
            wCoefsList[i] = (i, wCoefsList[i] / len2)
   
    print 'wCoefsList: ', wCoefsList
    return sc.parallelize(wCoefsList)

# ------------- end of punkt 2 -----------------
# ------------- punkt 3 ------------------------
def multPunkt3(record):
    featureNumber = record[0]
    ID, value1 = record[1][0]
    value2 = record[1][1]
    if value1 == None:
        value1 = 0.0
    if value2 == None:
        value2 = 0.0
    return (ID, value1*value2)

def generateResultsFromWCoefs(trainRDD, wCoefsRDD, parameters):
    tmpJoinRDD = trainRDD.join(wCoefsRDD)
    #print 'tmpJoinRDD.takeSample(10) = ', tmpJoinRDD.takeSample(False, 10, 57)
    tmpJoinRDD = tmpJoinRDD.map(multPunkt3)
    #print 'tmpJoinRDD.takeSample(10) = ', tmpJoinRDD.takeSample(False, 10, 57)
    return tmpJoinRDD.reduceByKey(aggregateSum) 
# ------------- end of punkt 3 -----------------
# ------------- punkt 3.5 ----------------------
def getClickResultsRDD(trainClickRDD, resultsRDD):
    return trainClickRDD.join(resultsRDD)
# ------------- end of punkt 3.5 ---------------
# ------------- punkt 4 ------------------------
def calcLogit(record):
    #print "LBFGS: calcLogit: record = "
    value = -record[1][0]*record[1][1]
    if value > 20.0:
        return (record[0], value)
    return (record[0], math.log(1.0 + math.exp(value)))    

def makeSquareValue(record):
    return (record[0], record[1] * record[1])

def getLogisticLoss(clickResultsRDD, wCoefsRDD, parameters):
    #print "LBFGS: getLogisticLoss started"
    #print "LBFGS: clickResultsRDD.count() = ", clickResultsRDD.count()
    #print "LBFGS: clickResultsRDD.takeSample(10) = ", clickResultsRDD.takeSample(False, 10, 179)
    tmp = clickResultsRDD.map(calcLogit);
    #print "LBFGS: tmp.count() = ", tmp.count()
    #ans = tmp.reduce(lambda x, y: (x[0], x[1] + y[1]))
    ans = tmp.reduce(aggregateSumValues)
    #print "LBFGS: getLogisticLoss: ans = ", ans
    ans = ans[1]
    if parameters.l2WeightDecay != 0.0:
       wCoefsSquareRDD = wCoefsRDD.map(makeSquareValue)
       wLength = wCoefsSquareRDD.reduce(aggregateSumValues)[1]
       #print 'getLogisticLoss: parameters=',parameters.l2WeightDecay,', wLength=',wLength
       #print type(parameters.l2WeightDecay)
       ans1 = parameters.l2WeightDecay * 0.5 
       ans1 = ans1 * wLength
       ans += ans1
    #print "LBFGS: getLogisticLoss: ans = ", ans

    return ans
# ------------- end of punkt 4 -----------------
# ------------- punkt 5 ------------------------
def swapNumFAndID(record):
    return record[1][0], (record[0], record[1][1])
    
def calcLogLossGradientSummand(record):
    ID, ((numf, vf), (y, ys)) = record
    if y*ys > 20:
        return numf, -vf * y * math.exp(-y*ys)
    return numf, -vf * y / (1.0 + math.exp(y*ys))

def summatorWithCoef(coef):
    def f(value):
        #print 'VALUE ', value
        ans = value[1][0]
        if ans == None:
            ans = 0.0
        if value[1][1] != None:
            ans += coef * value[1][1]
        return value[0], ans
    return f

def calcLogLossGradient(trainRDD, clickResultsRDD, wCoefs, parameters):
    tmpTrainRDD = trainRDD.map(swapNumFAndID)
    tmpRDD = tmpTrainRDD.join(clickResultsRDD)
    gradRDD = tmpRDD.map(calcLogLossGradientSummand).reduceByKey(aggregateSum)
    if parameters.l2WeightDecay != 0.0:
        gradRDD = gradRDD.join(wCoefs).map(summatorWithCoef(parameters.l2WeightDecay))    
    return gradRDD
# ------------- end of punkt 5 -----------------
## ------------- punkt 6 ------------------------
def calcLogLossGradientTwiceSummand(record):
    ID, ((numf, vf), (y, ys)) = record
    if y*ys > 20:
        return (numf, vf*vf*y*y * math.exp(-y*ys))
    expyy = math.exp(y*ys)
    return (numf, vf * vf * y * y * expyy / ((1.0 + expyy) ** 2))
    #return numf, vf * vf * y * y / (1.0 + math.exp(-y*ys))

def calcLogLossTwiceGradient(trainRDD, clickResultsRDD, wCoefs, parameters):
    tmpTrainRDD = trainRDD.map(swapNumFAndID)
    tmpRDD = tmpTrainRDD.join(clickResultsRDD)
    gradRDD = tmpRDD.map(calcLogLossGradientTwiceSummand).reduceByKey(aggregateSum)
    if parameters.l2WeightDecay != 0.0:
        gradRDD = gradRDD.map(lambda x: (x[0], x[1] + parameters.l2WeightDecay))    
    return gradRDD
# ------------- end of punkt 6 -----------------
# ------------- punkt 7 ------------------------
def multInValue(record):
    v0 = record[1][0];
    if v0 == None:
        v0 = 0.0
    v1 = record[1][1];
    if v1 == None:
        v1 = 0.0
    
    return record[0], v0*v1

def scalMul(wCoefs1RDD, wCoefs2RDD):
    #print 'scalMul started'
    #print wCoefs1RDD.takeSample(False, 10, 49)
    #print wCoefs2RDD.takeSample(False, 10, 49)
    tmpJoinRDD = wCoefs1RDD.join(wCoefs2RDD)
    #print tmpJoinRDD.takeSample(False, 10, 49)
    tmpJoinRDD = tmpJoinRDD.map(multInValue)
    #print tmpJoinRDD.takeSample(False, 10, 49)
    #print 'tmpJoinRDD len: ', tmpJoinRDD.count()
    ans = tmpJoinRDD.reduce(aggregateSumValues)
    #print 'scalMul result: ans(key, value): ', ans
    return ans[1]
# ------------- end of punkt 7 -----------------
# ------------- calcPoints ---------------------

def calcDiff(record):
    y = record[1][0]
    yy = record[1][1]
    if yy == 0.0:
        return (0, 1, 0)
    if yy > 0.0:
        yy = 1
    else:
        yy = -1
    if y == yy:
        return (0, 0, 1)
    else:
        return (1, 0, 0)

def sum3(r1, r2):
    return r1[0]+r2[0], r1[1]+r2[1], r1[2]+r2[2]

def calcPoints(clickResultsRDD):
    return clickResultsRDD.map(calcDiff).reduce(sum3)  
# -------------- end of calcPoints ------------

# -------------- LBFGS Matrix -----------------
def localF(a, b):
    def f(x):
        return (x[0], x[1][0]*a + x[1][1]*b)
    return f

class Matrix:
    def __init__(self, a, b, gRDD, wRDD):
       self.a = a
       self.b = b
       self.gRDD = gRDD
       self.wRDD = wRDD

    def transpose(self):
        return Matrix(self.a, self.b, self.wRDD, self.gRDD)

    def multOnVector(self, vRDD):
        vw = scalMul(self.wRDD, vRDD) * self.b
        #print 'vw = ', vw
        if self.a == 0.0:
            return self.gRDD.map(lambda x: (x[0], x[1] * vw))
        else:
            #print 'self.a = ', self.a
            ans = vRDD.join(self.gRDD)
            #print 'Matrix: vRDDjoined.takeSample(10) = ', ans.takeSample(False, 10, 49)
            #ans = ans.map(lambda x: (x[0], getValueOr0(x[1][0]) * self.a + getValueOr0(x[1][1]) * vw))
            #ans = ans.map(lambda x: (x[0], getValueOr0(x[1][0])))
            ans = ans.map(localF(self.a, vw))
            #print 'Matrix: ansMapped = ', ans.takeSample(False, 10, 2340)
            return ans
# ------- end of LBFGS Matrix -----------------

# -------------- LBFGS DiagMatrix -------------
class DiagMatrix:
    def __init__(self, diagRDD):
        self.diagRDD = diagRDD

    def multOnVector(self, vRDD):
        #print 'self.diagRDD.takeSample(10)', self.diagRDD.takeSample(False, 10, 34)
        #print 'vRDD: ', vRDD
        #print 'vRDD,takeSample(10)', vRDD.takeSample(False, 10, 34)
        ans = self.diagRDD.join(vRDD)
        #print 'ans.takeSample(10)', ans.takeSample(False, 10, 34)
        return ans.map(lambda x: (x[0], getValueOr0(x[1][0]) * getValueOr0(x[1][1])))
# ------- end of LBFGS DiagMatrix -------------
# ------- LBFGS NextStepMatrix ----------------
class NextStepMatrix:
    def __init__(self, prevStepsNumber, d0):
        self.prevStepsNumber = prevStepsNumber
        self.fs = [d0]
        self.ss = []
        assert(self.prevStepsNumber >= 0)

    def addStepResults(self, fMatrix, sMatrix):
        self.fs.append(fMatrix)
        self.ss.append(sMatrix)
        while len(self.ss) > self.prevStepsNumber + 1:
            self.fs.pop(1)
            self.ss.pop(0)
        
    def multOnVector(self, vRDD):
        vRDDs = [vRDD]
        m = len(self.ss) - 1
        #print 'self.ss: ', self.ss
        #print 'self.fs: ', self.fs
        for i in range(m, -1, -1):
        #    print 'EEK vRDDs[-1]', vRDDs[-1].count()
        #    print 'EEK vRDDs.take(10)', vRDDs[-1].take(10)
            vRDDs.append(self.ss[i].multOnVector(vRDDs[-1]))
        #    print 'EEK self.ss[i]', self.ss[i]
        #    print 'EEK vRDDs[-1]', vRDDs[-1].take(10)
        
        for i in range(0, m+2):
        #    print 'EEK vRDDs[i]', vRDDs[i].count()
            vRDDs[i] = self.fs[m+1-i].multOnVector(vRDDs[i])
        ans = vRDDs[m+1]
        for i in range(m, -1, -1):
            ans = self.ss[m-i].multOnVector(ans).join(vRDDs[i]).map(lambda x: (x[0], getValueOr0(x[1][0]) + getValueOr0(x[1][1])))
        return ans
# ------- end of LBFGS NextStepMatrix ---------



class ResLossInfo:
    def __init__(self, trainRDD, trainClickRDD, wCoefsRDD, parameters):
        #print 'LBFGS: ResLossInfo constructor'
        self.resultsRDD = generateResultsFromWCoefs(trainRDD, wCoefsRDD, parameters)
        #print 'LBFGS: self.resultsRDD.takeSample(10) = ', self.resultsRDD.takeSample(False, 10, 2007)
        #print 'LBFGS: self.resultsRDD.count() = ', self.resultsRDD.count()
        #print 'LBFGS: ResLossInfo.resultsRDD'
        self.clickResultsRDD = getClickResultsRDD(trainClickRDD, self.resultsRDD)
        #print 'LBFGS: self.clickResults.takeSample(10) = ', self.clickResultsRDD.takeSample(False, 10, 2007)
        #print 'LBFGS: self.clickResults.count() = ', self.clickResultsRDD.count()
        #print 'LBFGS: ResLossInfo.clickResultsRDD'
        #self.gradRDD = calcLogLossGradient(trainRDD, clickResultsRDD, wCoefs, parameters)
        self.loss = getLogisticLoss(self.clickResultsRDD, wCoefsRDD, parameters)
#        if parameters.verbose:
#            print 'LBFGS: self.loss = ', self.loss
        #print 'LBFGS: ResLossInfo.loss'

    
def splitPair(x):
    f = x.split(',')
    return (f[0], int(f[1]) * 2 - 1)

def deleteFileOrDirectory(name):
    if os.path.exists(name):
        if (os.path.isfile(name)):
            os.remove(name)
        else:
            shutil.rmtree(name)
    
def mainLBFGS(parameters):
    sc = LocalSparkContext()
    #print 'LBFGS: MAIN STARTED'
    #print 'LBFGS: parameters initialized'
    #inputRDD = sc.textFile('train.txt')
    #inputRDD = sc.textFile('trainFull/trainFactorsSmall.txt')
    #inputTestRDD = sc.textFile('trainFull/testFactorsSmall.txt')
    inputRDD = sc.textFile(parameters.inputTrainTable)
    inputTestRDD = sc.textFile(parameters.inputTestTable)
    #inputRDD = parseFromTextFile(parameters.inputTrainTable)
    #inputTestRDD = parseFromTextFile(parameters.inputTestTable)

    ##print inputRDD
    #print 'LBFGS: inputRDD.count() = ', inputRDD.count()
    #print 'LBFGS: inputRDD initialized'
    trainClickRDD = makeIDClickRDD(inputRDD, parameters)
    #return None
    
    #testClickRDD = sc.textFile('trainFull/testResults.txt').map(splitPair)
    #print 'LBFGS: trainClickRDD.takeSample(10) = ', trainClickRDD.takeSample(False, 10, 218)
    #print 'LBFGS: trainClickRDD.count() = ', trainClickRDD.count()
    #print 'LBFGS: testClickRDD.count() = ', testClickRDD.count()
    if parameters.verbose:
        print 'LBFGS: trainClickRDD initialized'
    trainRDD = makeFeaturesValuesRDD(inputRDD, parameters, True)
    testRDD = makeFeaturesValuesRDD(inputTestRDD, parameters, False)
    #print 'LBFGS: trainRDD.takeSample(10) = ', trainRDD.takeSample(False, 10, 218)
    #print 'LBFGS: trainRDD.count() = ', trainRDD.count()
    #print 'LBFGS: testRDD.count() = ', testRDD.count()
    if parameters.verbose:
        print 'LBFGS: trainRDD initialized'
    #wCoefsRDD = generateWCoefsInit(sc, trainRDD, parameters)
    wCoefsRDD = generateWCoefsStochasticGradient(sc, parameters)
    print 'LBFGS: wCoefs.takeSample(10) = ', wCoefsRDD.takeSample(False, 10, 218)
#    return None
    print 'LBFGS: wCoefsRDD.count() = ', wCoefsRDD.count()
    if parameters.verbose:
        print 'LBFGS: wCoefsRDD initialized'
    resLossInfo = ResLossInfo(trainRDD, trainClickRDD, wCoefsRDD, parameters)
    #resLossTestInfo = ResLossInfo(testRDD, testClickRDD, wCoefsRDD, parameters)
    if parameters.verbose:
        print 'LBFGS: ResLossInfo initialized'
       
    twiceGradientMatrixRDD = calcLogLossTwiceGradient(trainRDD, resLossInfo.clickResultsRDD, wCoefsRDD, parameters)
    #print 'LBFGS: twiceGradientMatrixRDD.takeSample(10) = ', twiceGradientMatrixRDD.takeSample(False, 10, 218)
#    return None
    #print 'LBFGS: twiceGradientMatrixRDD.count() = ', twiceGradientMatrixRDD.count()
    
    twiceGradientMatrixRDD = twiceGradientMatrixRDD.map(lambda x: (x[0], 1.0 / x[1]))
    #twiceGradientMatrixRDD = twiceGradientMatrixRDD.map(lambda x: (x[0], 1.0))
    

    #twiceGradientMatrixRDD = twiceGradientMatrixRDD.map(lambda x: (x[0], 1.0 / x[1]))
    nextStepMatrix = NextStepMatrix(parameters.historySize,
                                                   DiagMatrix(twiceGradientMatrixRDD) 
                                   )
    if parameters.verbose:
        print 'LBFGS: wCoefsRDD.count() = ', wCoefsRDD.count()
        print 'LBFGS: nextStepMatrix initialized'
    
    #gradRDD = calcLogLossGradient(trainRDD, resLossInfo.clickResultsRDD, wCoefsRDD, parameters)
    if parameters.verbose:
        print 'LBFGS: gradRDD calculated'
    #naprRDD = nextStepMatrix.multOnVector(gradRDD)
    if parameters.verbose:
        print 'LBFGS: naprRDD calculated'
        print 'LBFGS ITER 0: train points ', calcPoints(resLossInfo.clickResultsRDD)
     #   print 'LBFGS ITER 0: test points ', calcPoints(resLossTestInfo.clickResultsRDD)
    
    #preNewWCoefsRDD = wCoefsRDD.fullOuterJoin(naprRDD)
    gradRDD = calcLogLossGradient(trainRDD, resLossInfo.clickResultsRDD, wCoefsRDD, parameters)

    if parameters.verbose:
        print 'LBFGS: Gradient calculated'
        print 'LBFGS: start iterations'
    startAlpha = parameters.startAlpha

    scwg = 1.0
    for i in range(0, parameters.iterationsNumber):            
        #print 'LBFGS: gradRDD calculated'
        if parameters.verbose: 
            print 'Start iteration ', i
        naprRDD = nextStepMatrix.multOnVector(gradRDD)
        if parameters.verbose: 
            print 'napr found'

        #naprLen = naprRDD.map(lambda x: x[1]*x[1]).reduce(lambda x, y: x + y)
        #naprLen = math.sqrt(naprLen)
        #if naprLen < parameters.minDirectionVectorLength and parameters.verbose:
        #    print 'END OF ITERATIONS: direction vector length is ', naprLen

        preNewWCoefsRDD = wCoefsRDD.join(naprRDD)
        if parameters.verbose: 
            print 'preNewWCoefs found'
        
        alpha = parameters.alphaDivisor * parameters.startAlpha
        newWCoefsRDD = None
        newResLossInfo = None

        #print 'alpha = ', alpha
        scgd = scalMul(gradRDD, naprRDD)
        #print 'scgd = ', scgd
        while True:
            alpha = alpha / parameters.alphaDivisor
         #   if parameters.verbose:
         #       print 'ALPHA: ', alpha
            if alpha < parameters.finishAlpha:
                break
            #assert alpha > 0.000000001, 'TOO SMALL ALPHA!!!'
            newWCoefsRDD = preNewWCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][0]) - alpha * getValueOr0(x[1][1])))
            #newWCoefsRDD = preNewWCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][0]) + alpha * getValueOr0(x[1][1])))
            #print 'newWCoefsRDD.takeSample(10) = ', newWCoefsRDD.takeSample(False, 10, 32)
            #print 'newWCoefsRDD.count() = ', newWCoefsRDD.count()
            newResLossInfo = ResLossInfo(trainRDD, trainClickRDD, newWCoefsRDD, parameters)
         #   if parameters.verbose:
         #       print 'CURRENT LOSS: ', resLossInfo.loss
                #print 'POSSIBLY NEW LOSS: ', newResLossInfo.loss
                #print 'UPDATED LOSS: ', resLossInfo.loss + 0.9 * alpha * scgd
            #if resLossInfo.loss + 0.9 * alpha * scgd > newResLossInfo.loss:
            if resLossInfo.loss > newResLossInfo.loss:
                break
        
            newWCoefsRDD = preNewWCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][0]) + alpha * getValueOr0(x[1][1])))
            newResLossInfo = ResLossInfo(trainRDD, trainClickRDD, newWCoefsRDD, parameters)
            if resLossInfo.loss > newResLossInfo.loss:
                break
        
        if alpha < parameters.finishAlpha:
            if parameters.verbose:
                print 'END OF ITERATIONS: alpha is less than ', parameters.finishAlpha
            break

        startAlpha = alpha 
        newGradRDD = calcLogLossGradient(trainRDD, newResLossInfo.clickResultsRDD, newWCoefsRDD, parameters)
       # print 'newGradRDD.takeSample(10) = ', newGradRDD.takeSample(False, 10, 32)
        dwCoefsRDD = wCoefsRDD.join(newWCoefsRDD) 
       # print 'dwCoefsRDD.takeSample(10) = ', dwCoefsRDD.takeSample(False, 10, 32)
        dwCoefsRDD = dwCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][1]) - getValueOr0(x[1][0])))
       # print 'dwCoefsRDD.takeSample(10) = ', dwCoefsRDD.takeSample(False, 10, 32)
        dgRDD = gradRDD.join(newGradRDD).map(lambda x: (x[0], getValueOr0(x[1][1]) - getValueOr0(x[1][0])))
       # print 'dgRDD.takeSample(10) = ', dgRDD.takeSample(False, 10, 32)
        
        scwg = scalMul(dgRDD, dwCoefsRDD)
        #if parameters.verbose:
        #    print 'scwg = ', scwg
        assert not(scwg == 0.0), 'SCWG IS 0!!!'
        scwg = 1.0 / scwg
        nextStepMatrix.addStepResults(Matrix(0.0, scwg, dwCoefsRDD, dwCoefsRDD), Matrix(1.0, -scwg, dgRDD, dwCoefsRDD))
        wCoefsRDD = newWCoefsRDD
        resLossInfo = newResLossInfo
        if parameters.verbose:
            print 'LOSS: ',resLossInfo.loss
        gradRDD = newGradRDD
        #naprRDD = nextStepMatrix.multOnVector(gradRDD)
        #resLossTestInfo = ResLossInfo(testRDD, testClickRDD, wCoefsRDD, parameters)
        if parameters.verbose:
            print 'ITER ',i+1,  ': train points ', calcPoints(resLossInfo.clickResultsRDD)
      #      print 'ITER ',i+1,  ': test points ', calcPoints(resLossTestInfo.clickResultsRDD)

#    finalResultsRDD = generateResultsFromWCoefs(testRDD, wCoefsRDD, parameters)
    
    print wCoefsRDD.table
    if parameters.verbose:
        print 'Iterations finished. Generating and saving final prediction result...'
    finalResultsRDD = generateResultsFromWCoefs(testRDD, wCoefsRDD, parameters)
    #print 'LAST: ', finalResultsRDD.collect()
    finalResultsRDD = finalResultsRDD.map(lambda x: str(x[0]) + parameters.delimiter + str((sign(x[1]) + 1) / 2))
    
    #resultTableName = parameters.resultTestTable + 'Table'
    #deleteFileOrDirectory(parameters.resultTestTable)
    #deleteFileOrDirectory(resultTableName)
    
    finalResultRDD = finalResultsRDD.collect()
    #print finalResultRDD
    #finalResultsRDD.saveAsTextFile(parameters.resultTestTable)
    
    f = open(parameters.resultTestTable, 'w')
    f.write('ID' + parameters.delimiter + 'CLICK\n')
    for v in finalResultRDD:
        f.write(v + '\n')
    f.close()
    #for fpart in os.listdir(resultTableName):
    #    if re.match('part', fpart) is not None:
    #        ff = open(resultTableName + '/' + fpart, 'r')
    #        for line in ff:
    #            f.write(line)
    #        ff.close()
    #f.close()
    #deleteFileOrDirectory(resultTableName)

class Parameters:
    def __init__(self):
    #    self.catFeatures = 37
    #    self.numFeatures = 23
    #    self.l2WeightDecay = 1.0
    #    self.historySize = 4
    #    self.iterationsNumber = 2
        #self.startAlpha = 1.0
        #self.finishAlpha = 0.000000001
        #self.alphaDivisor = 2
        #self.minDirectionVectorLength = 0.000000001
        #self.verbose = True
        self.catFeatureVariants = 2 ** 17
        #self.inputTrainTable = 'trainFull/trainFactorsSmall.txt'
        #self.inputTestTable = 'trainFull/testFactorsSmall.txt'
        #self.resultTestTable = 'trainFull/testResultSmall.txt'
        #self.delimiter = ','

def generateParser():
    parser = argparse.ArgumentParser(description='LBFGS parameters')
    parser.add_argument('-i', '--inputTrainTable', help='Name of train table file')
    parser.add_argument('-it', '--inputTestTable', help='Name of test table file')
    parser.add_argument('-o', '--resultTestTable', help='Name of file to save result predictions')

    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of features in one line in tables')
    parser.add_argument('-cf', '--catFeatures', type=int, help='Number of categorial features')
    parser.add_argument('-nf', '--numFeatures', type=int, help='Number of numeric features')
    parser.add_argument('-m', '--historySize', type=int, default=4, help='Number of previous steps using in current iteration')
    parser.add_argument('-iters', '--iterationsNumber', type=int, default=20, help='Maximal number of iterations to do')
    parser.add_argument('-cfv', '--catFeatureVariants', type=int, default = 2 ** 17, help='how much variables for catFeatureVariants is possible')

    parser.add_argument('-sa', '--startAlpha', type=float, default = 1.0, help='maximal alpha')
    parser.add_argument('-fa', '--finishAlpha', type=float, default = 1e-9, help='minimal alpha')
    parser.add_argument('-stepa', '--alphaDivisor', type=float, default = 2, help='divisor of alpha (if current alpha didn\'t decrease loss function)')
    parser.add_argument('-v', '--verbose', type=bool, default = False, help='comments about current state of work')

    parser.add_argument('-l2', '--l2WeightDecay', type=float, default=0.0, help='Coefficient l2-regularization')
    parser.add_argument('-hdfs', '--hdfs', type=bool, default=False, help='true if files are in HDFS-system')

    return parser 

def main():
    parser = generateParser()
    #parameters = Parameters()
    print 'sys.argv=', sys.argv
    args = sys.argv
    args.pop(0)
    print args
    parameters = parser.parse_args(args)
    mainLBFGS(parameters)


#a = [(1, 0), (4, 1), (1, 4)]
#sc = LocalSparkContext()
#table = sc.parallelize(a)
#print table.table
#table = table.reduceByKey(aggregateSum)
#print table.table

print 'LBFGS learning started'
main()
print 'LBFGS learning finished'
