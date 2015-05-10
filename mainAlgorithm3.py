#from pyspark import SparkContext
import math, sys, os, shutil, argparse, re, random

#------------------------- spark emulation procedures -----------
def findNextPositionKeyNonEqual(lst, index):
    while index + 1 < len(lst) and lst[index][0] == lst[index + 1][0]:
        index += 1
    return index + 1

class LocalTable:
    def __init__(self, table):
        self.table = table

    def map(self, f):
        newTable = []
        for record in self.table:
            newTable.append(f(record))
        return LocalTable(newTable)

    def flatMap(self, f):
        newTable = []
        for record in self.table:
            lst = f(record)
            for recordLst in lst:
                newTable.append(recordLst)
        return LocalTable(newTable)

    def join(self, table):
        lst1 = self.table
        lst1.sort()
        lst2 = table.table
        lst2.sort()

        index1 = 0
        index2 = 0
        answer = []
        while (index1 < len(lst1) and index2 < len(lst2)):
            if lst1[index1][0] < lst2[index2][0]:
                index1 += 1
            elif lst1[index1][0] > lst2[index2][0]:
                index2 += 1
            else:
                nextIndex1 = findNextPositionKeyNonEqual(lst1, index1)
                nextIndex2 = findNextPositionKeyNonEqual(lst2, index2)
                while index1 < nextIndex1:
                    for i in range(index2, nextIndex2):
                        answer.append((lst1[index1][0], (lst1[index1][1], lst2[i][1])))
                    index1 += 1
                index2 = nextIndex2
        return LocalTable(answer)

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
        return LocalTable(answer)

    def reduce(self, f):
        record = self.table[0]
        for index in range(1, len(self.table)):
            record = f(record, self.table[index])
        return record

    def collect(self):
        return self.table

class LocalContext:
    def textFile(self, fileName):
        ans = []
        f = open(fileName, 'r')
        for line in f:
            ans.append(line[:-2])
        return LocalTable(ans)

    def parallelize(self, lst):
        return LocalTable(lst)

#----------------- end of spark emulation procedures ------------
#-----------parsing of input tables and building tables ---------

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
               
#--- end of parsing of input tables and building tables ---------
# ------------- punkt 0 ------------------------
def makeIDClick(parameters, isIDHash):
    def f(strValue):
        return makeIDClickOrFeaturesValues(parameters, isIDHash, strValue, True)
    return f

def makeIDClickRDD(trainInputRDD, parameters, isIDHash = True):
    f = makeIDClick(parameters, isIDHash)
    trainClickRDD = trainInputRDD.flatMap(f)
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
#def generateWCoefsInit(sc, trainRDD, parameters):
#    wCoefsRDD = trainRDD.map(lambda x: (x[0], 0.0)).reduceByKey(lambda x, y: x)
#    return wCoefsRDD

def generateWCoefsStochasticGradient(sc, parameters):
    wCoefsList = [0.0] * (1 + parameters.catFeatureVariants * parameters.catFeatures + parameters.numFeatures)
    f = open( parameters.inputTrainTable, 'r')
    nu = 0.1
    for line in f:
        line = line[0:len(line)-1]
        featuresList = makeIDClickOrFeaturesValues(parameters, False, line, False)
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
        len2 +=  (wCoefsList[i] ** 2)
    
    len2 = len2 ** 0.5
    if len2 != 0.0:
        for i in range(0, len(wCoefsList)):
            wCoefsList[i] = (i, wCoefsList[i] / len2)
   
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
    tmpJoinRDD = tmpJoinRDD.map(multPunkt3)
    return tmpJoinRDD.reduceByKey(lambda x, y: x+y)
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
    tmp = clickResultsRDD.map(calcLogit);
    ans = tmp.reduce(aggregateSumValues)
    ans = ans[1]
    if parameters.l2WeightDecay != 0.0:
       wCoefsSquareRDD = wCoefsRDD.map(makeSquareValue)
       wLength = wCoefsSquareRDD.reduce(aggregateSumValues)[1]
       #print 'getLogisticLoss: parameters=',parameters.l2WeightDecay,', wLength=',wLength
       #print type(parameters.l2WeightDecay)
       ans1 = parameters.l2WeightDecay * 0.5 
       ans1 = ans1 * wLength
       ans += ans1

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
    tmpJoinRDD = wCoefs1RDD.join(wCoefs2RDD)
    tmpJoinRDD = tmpJoinRDD.map(multInValue)
    ans = tmpJoinRDD.reduce(aggregateSumValues)
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
        if self.a == 0.0:
            return self.gRDD.map(lambda x: (x[0], x[1] * vw))
        else:
            ans = vRDD.join(self.gRDD)
            ans = ans.map(localF(self.a, vw))
            return ans
# ------- end of LBFGS Matrix -----------------

# -------------- LBFGS DiagMatrix -------------
class DiagMatrix:
    def __init__(self, diagRDD):
        self.diagRDD = diagRDD

    def multOnVector(self, vRDD):
        ans = self.diagRDD.join(vRDD)
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
        for i in range(m, -1, -1):
            vRDDs.append(self.ss[i].multOnVector(vRDDs[-1]))
        
        for i in range(0, m+2):
            vRDDs[i] = self.fs[m+1-i].multOnVector(vRDDs[i])
        ans = vRDDs[m+1]
        for i in range(m, -1, -1):
            ans = self.ss[m-i].multOnVector(ans).join(vRDDs[i]).map(lambda x: (x[0], getValueOr0(x[1][0]) + getValueOr0(x[1][1])))
        return ans
# ------- end of LBFGS NextStepMatrix ---------



class ResLossInfo:
    def __init__(self, trainRDD, trainClickRDD, wCoefsRDD, parameters):
        self.resultsRDD = generateResultsFromWCoefs(trainRDD, wCoefsRDD, parameters)
        self.clickResultsRDD = getClickResultsRDD(trainClickRDD, self.resultsRDD)
        self.loss = getLogisticLoss(self.clickResultsRDD, wCoefsRDD, parameters)

    
def splitPair(x):
    f = x.split(',')
    return (f[0], int(f[1]) * 2 - 1)

#def deleteFileOrDirectory(name):
#    if os.path.exists(name):
#        if (os.path.isfile(name)):
#            os.remove(name)
#        else:
#            shutil.rmtree(name)
    
def mainLBFGS(parameters):
    sc = LocalContext()
    inputRDD = sc.textFile(parameters.inputTrainTable)
    inputTestRDD = sc.textFile(parameters.inputTestTable)
    trainClickRDD = makeIDClickRDD(inputRDD, parameters)
    if parameters.verbose:
        print 'trainClickRDD initialized'
    trainRDD = makeFeaturesValuesRDD(inputRDD, parameters, True)
    testRDD = makeFeaturesValuesRDD(inputTestRDD, parameters, False)
    if parameters.verbose:
        print 'trainRDD initialized'
    
    wCoefsRDD = generateWCoefsStochasticGradient(sc, parameters)
    if parameters.verbose:
        print 'wCoefsRDD initialized'
    
    resLossInfo = ResLossInfo(trainRDD, trainClickRDD, wCoefsRDD, parameters)
    if parameters.verbose:
        print 'ResLossInfo initialized'
       
    twiceGradientMatrixRDD = calcLogLossTwiceGradient(trainRDD, resLossInfo.clickResultsRDD, wCoefsRDD, parameters)
    
    twiceGradientMatrixRDD = twiceGradientMatrixRDD.map(lambda x: (x[0], 1.0 / x[1]))
    nextStepMatrix = NextStepMatrix(parameters.historySize,
                                                   DiagMatrix(twiceGradientMatrixRDD) 
                                   )
    if parameters.verbose:
        print 'wCoefsRDD.count() = ', wCoefsRDD.count()
        print 'nextStepMatrix initialized'
    
    if parameters.verbose:
        print 'gradRDD calculated'
    if parameters.verbose:
        print 'directionRDD calculated'
        print 'ITER 0: train points ', calcPoints(resLossInfo.clickResultsRDD)
        print 'LOSS: ', resLossInfo.loss
    
    gradRDD = calcLogLossGradient(trainRDD, resLossInfo.clickResultsRDD, wCoefsRDD, parameters)

    if parameters.verbose:
        print 'Start gradient calculated'
    startAlpha = parameters.startAlpha

    scwg = 1.0
    for i in range(0, parameters.iterationsNumber):            
        if parameters.verbose: 
            print 'Start iteration ', i
        directionRDD = nextStepMatrix.multOnVector(gradRDD)
        if parameters.verbose: 
            print 'direction found'

        preNewWCoefsRDD = wCoefsRDD.join(directionRDD)
        if parameters.verbose: 
            print 'preNewWCoefs found'
        
        alpha = parameters.alphaDivisor * parameters.startAlpha
        newWCoefsRDD = None
        newResLossInfo = None

        scgd = scalMul(gradRDD, directionRDD)
        while True:
            alpha = alpha / parameters.alphaDivisor
            if alpha < parameters.finishAlpha:
                break
            for signum in [-1, +1]:
                newWCoefsRDD = preNewWCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][0]) + signum * alpha * getValueOr0(x[1][1])))
                newResLossInfo = ResLossInfo(trainRDD, trainClickRDD, newWCoefsRDD, parameters)
                if resLossInfo.loss > newResLossInfo.loss:
                    break
            
            if resLossInfo.loss > newResLossInfo.loss:
                break

            
        if alpha < parameters.finishAlpha:
            if parameters.verbose:
                print 'END OF ITERATIONS: alpha is less than ', parameters.finishAlpha
            break

        startAlpha = alpha 
        newGradRDD = calcLogLossGradient(trainRDD, newResLossInfo.clickResultsRDD, newWCoefsRDD, parameters)
        dwCoefsRDD = wCoefsRDD.join(newWCoefsRDD) 
        dwCoefsRDD = dwCoefsRDD.map(lambda x: (x[0], getValueOr0(x[1][1]) - getValueOr0(x[1][0])))
        dgRDD = gradRDD.join(newGradRDD).map(lambda x: (x[0], getValueOr0(x[1][1]) - getValueOr0(x[1][0])))
        
        scwg = scalMul(dgRDD, dwCoefsRDD)
        assert not(scwg == 0.0), 'SCWG IS 0!!!'
        scwg = 1.0 / scwg
        nextStepMatrix.addStepResults(Matrix(0.0, scwg, dwCoefsRDD, dwCoefsRDD), Matrix(1.0, -scwg, dgRDD, dwCoefsRDD))
        wCoefsRDD = newWCoefsRDD
        resLossInfo = newResLossInfo
        gradRDD = newGradRDD
        if parameters.verbose:
            print 'ITER ',i+1,  ': train points ', calcPoints(resLossInfo.clickResultsRDD)
            print 'LOSS: ',resLossInfo.loss

    if parameters.verbose:
        print 'Iterations finished. Generating and saving final prediction result...'
    finalResultsRDD = generateResultsFromWCoefs(testRDD, wCoefsRDD, parameters)
    finalResultsRDD = finalResultsRDD.map(lambda x: str(x[0]) + parameters.delimiter + str((sign(x[1]) + 1) / 2))
    
    finalResultRDD = finalResultsRDD.collect()
    
    f = open(parameters.resultTestTable, 'w')
    f.write('ID' + parameters.delimiter + 'CLICK\n')
    for v in finalResultRDD:
        f.write(v + '\n')
    f.close()

def generateParser():
    parser = argparse.ArgumentParser(description='LBFGS parameters')
    parser.add_argument('-i', '--inputTrainTable', help='Name of train table file')
    parser.add_argument('-it', '--inputTestTable', help='Name of test table file')
    parser.add_argument('-o', '--resultTestTable', help='Name of file to save result predictions')

    parser.add_argument('-m', '--historySize', type=int, default=4, help='Number of previous steps using in current iteration (4 by default)')
    parser.add_argument('-cf', '--catFeatures', type=int, help='Number of categorial features')
    parser.add_argument('-nf', '--numFeatures', type=int, help='Number of numeric features')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of features in one line in tables (comma by default)')
    parser.add_argument('-iters', '--iterationsNumber', type=int, default=20, help='Maximal number of iterations to do(20 by default)')
    parser.add_argument('-cfv', '--catFeatureVariants', type=int, default = 2 ** 17, help='how much variables for catFeatureVariants is possible (2 ** 17 by default)')

    parser.add_argument('-sa', '--startAlpha', type=float, default = 1.0, help='maximal alpha (1.0 by default)')
    parser.add_argument('-fa', '--finishAlpha', type=float, default = 1e-18, help='minimal alpha (1e-18 by default)')
    parser.add_argument('-stepa', '--alphaDivisor', type=float, default = 2, help='divisor of alpha (if current alpha didn\'t decrease loss function; 2 by default)')
    parser.add_argument('-v', '--verbose', type=bool, default = False, help='comments about current state of work(True or False; False by default)')
    parser.add_argument('-l2', '--l2WeightDecay', type=float, default=0.0, help='Coefficient l2-regularization (0.0 by default)')


    return parser 

def main():
    parser = generateParser()
    args = sys.argv
    args.pop(0)
    parameters = parser.parse_args(args)
    assert parameters.alphaDivisor > 1.0, 'ERROR: alphaDivisor(stepa) should be strictly more than 1.0'
    mainLBFGS(parameters)

print 'LBFGS learning started'
main()
print 'LBFGS learning finished'
