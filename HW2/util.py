import pickle
import numpy as np
import matplotlib.pyplot as plt

mnist_data = {}
int_Train = []
int_TrainLbl = []
int_Dev = []
int_DevLbl = []
X = []
Y = []
testX = []
testY = []

def getDS():
  pkl_name = '../mnist_rowmajor.pkl'
  global mnist_data
  mnist_data = pickle.load( open( pkl_name, "rb" ) )


def getTrainData():
  return mnist_data['images_train']

def splitTrainSequential(splitPercent):
   imgTrain = mnist_data['images_train']
   lblTrain = mnist_data['labels_train']
   M = int(lblTrain.shape[0] * splitPercent)
   global int_Train,int_TrainLbl,int_Dev,int_DevLbl
   int_Train = imgTrain[:M,:]
   int_TrainLbl = lblTrain[:M,:]
   int_Dev = imgTrain[M:,:]
   int_DevLbl = lblTrain[M:,:]

def splitStratifiedTrainSequential(splitPercent):
   imgTrain = mnist_data['images_train']
   lblTrain = mnist_data['labels_train']
   inds = []
   inds2 = []
   for i in range(10):
     ind = np.where(lblTrain == i)[0]
     M = int(len(ind) * splitPercent)
     ind1 = ind[:M]
     ind2 = ind[M:]
     inds.extend(ind1)
     inds2.extend(ind2)
   inds.sort()
   inds2.sort()
   global int_Train,int_TrainLbl,int_Dev,int_DevLbl
   int_Train = imgTrain[inds,:]
   int_TrainLbl = lblTrain[inds,:]
   int_Dev = imgTrain[inds2,:]
   int_DevLbl = lblTrain[inds2,:]


def splitStratifiedTrainRandom(splitPercent):
   imgTrain = mnist_data['images_train']
   lblTrain = mnist_data['labels_train']
   inds = []
   inds2 = []
   for i in range(10):
     ind = np.where(lblTrain == i)[0]
     M = int(len(ind) * splitPercent)
     ind1 = np.random.choice(ind,M,replace=False)
     ind2 = list(set(ind) - set(ind1))
     inds.extend(ind1)
     inds2.extend(ind2)
   inds.sort()
   inds2.sort()
   global int_Train,int_TrainLbl,int_Dev,int_DevLbl
   int_Train = imgTrain[inds,:]
   int_TrainLbl = lblTrain[inds,:]
   int_Dev = imgTrain[inds2,:]
   int_DevLbl = lblTrain[inds2,:]


def testSplitTrainSequential(M,N):
   imgTrain = mnist_data['images_train']
   lblTrain = mnist_data['labels_train']
   global int_Train,int_TrainLbl,int_Dev,int_DevLbl
   int_Train = imgTrain[:M,:]
   int_TrainLbl = lblTrain[:M,:]
   int_Dev = imgTrain[M:M+N,:]
   int_DevLbl = lblTrain[M:M+N,:]

def countSplit():
   print "Int Train : ",
   for i in range(10):
       ind = np.where(int_TrainLbl == i)[0]
       print len(ind),
   print "\nInt Dev : ",
   for i in range(10):
       ind = np.where(int_DevLbl == i)[0]
       print len(ind),
   print "\nFull training List : ",
   for i in range(10):
       ind = np.where(mnist_data['labels_train'] == i)[0]
       print len(ind),

def histogram(msg):
  plt.figure(1)
  plt.subplot(121)
  plt.hist(int_TrainLbl, bins=[0, 1, 2, 3,4,5,6,7,8,9,10])
  plt.title("intTrain classes")
  plt.axis([0,10,0,5000])
  plt.subplot(122)
  plt.hist(int_DevLbl, bins=[0, 1, 2, 3,4,5,6,7,8,9,10])
  plt.title("intDev classes")
  plt.axis([0,10,0,5000])
  plt.show()
  plt.savefig(msg)

def getXY(dParm):
   global X,Y,testX,testY
   if dParm == 1 :
      X = int_Train
      Y = int_TrainLbl
      testX = int_Dev
      testY = int_DevLbl
   elif dParm == 2:
      X = mnist_data['images_train']
      Y = mnist_data['labels_train']
      testX = mnist_data['images_test']
      testY = mnist_data['labels_test']

def predictClass(W,x,b):
  W1 = np.dot(W,x) + b.T
  return np.argmax(W1)

def MLPtest(W,b):
   cNo = 0
   for indX in range(testX.shape[0]):
      x = testX[indX,:]
      y = testY[indX,:]
      yH = predictClass(W,x,b)
      if y == yH:
         cNo += 1
   acc = float(cNo) / float(len(testY))
   print "No of Matches = ", cNo, ", Accuracy = ",acc

def MLPtrain(mIter,lrType):
  lr = 0.01
  tO = 1
  k = 1
  W = np.zeros((10,X.shape[1]))
  b = np.zeros((10,1))
#  W = np.random.uniform(-1.0, 1.0,(10,X.shape[1]))
#  b = np.random.uniform(-1.0, 1.0,(10,1))
  for iter in range(mIter):
    if lrType == 1:
       lr = 0.01
    elif lrType == 2:
       lr = (iter + 1 + tO) ** -k
#    print "Learning rate : ",lr  
    if (iter + 1) % 100 == 0:
      print "Iteration :: " , iter + 1
    for indX in range(X.shape[0]):
      x = X[indX,:]
      y = Y[indX,:]
      yH = predictClass(W,x,b)
      if y != yH:
        W[y,:] += lr * x
        b[y] += lr * 1
        W[yH,:] -= lr * x
        b[yH] -= lr * 1
#    if (iter + 1) % 100 == 0:
#        MLPtest(W,b)
  return (W,b)

def sig(a):
  h = 1.0/(1.0 + np.exp(-1 * a))
  return h

def sigDerivative(a):
   d = a * (1.0 - a)
   return d


def NN(K,lr1,mIter,actParm,lrParm):
  lr = 0.1

  eps = 1.0 * 10 ** -8
  eta = 0.1

  if lrParm == 1:
    lr = lr1

  batch = 10
  D = X.shape[1]

  y = np.zeros((X.shape[0],10))
  for indX in range(X.shape[0]):
     y1 = Y[indX,:]
     y[indX,y1] = 1

  w1 = np.random.uniform(-1.0, 1.0,(D,K))
  w2 = np.random.uniform(-1.0, 1.0, (K,10))
  startB = 0

  pG1 = np.zeros((D,K))
  pG2 = np.zeros((K,10))

  for iter in xrange(mIter):
   while startB < X.shape[0]:
    if lrParm == 2:
      k1 = np.multiply(w1,w1)
      pG1 = pG1 + k1
      k2 = np.multiply(w2,w2)
      pG2 = pG2 + k2      

    endB = startB + batch
    if endB >= X.shape[0]:
     endB = X.shape[0]

    xB = X[startB:endB,:]
    yB = y[startB:endB,:]

    wX1 = np.dot(xB,w1)
    h1 = sig(wX1)
    wX2 = np.dot(h1,w2)
    h2 = sig(wX2)

    loss2 = yB - h2
    z2 = loss2 * sigDerivative(h2)
    loss1 = z2.dot(w2.T)
    z1 = loss1 * sigDerivative(h1)

    w2_2 = h1.T.dot(z2)
    w1_2 = xB.T.dot(z1)

    if lrParm == 1:
      w2 = w2 + lr * w2_2
      w1 = w1 + lr * w1_2
    elif lrParm == 2:
      lr1 = eta / (np.sqrt(eps + pG1))
      lr2 = eta / (np.sqrt(eps + pG2))      
      w2 = w2 + np.multiply(lr2,w2_2)
      w1 = w1 + np.multiply(lr1, w1_2)

    startB = endB
  return (w1,w2)

def NNTest(w1,w2):
   wX1 = np.dot(testX,w1)
   h1 = sig(wX1)
   wX2 = np.dot(h1,w2)
   h2 = sig(wX2)
   yH = [np.argmax(h2[i]) for i in range(h2.shape[0])]
   cAr = [1 for i in range(len(yH)) if yH[i] == testY[i]]
   cNo = np.sum(cAr)
   acc = float(cNo) / float(len(testY))
   print "No of Matches = ", str(cNo), ", Accuracy = ",str(acc)


def NNMLP(N,K,lr,mIter,actParm):
  batch = 100
  D = X.shape[1]

  y = np.zeros((X.shape[0],10))
  for indX in range(X.shape[0]):
     y1 = Y[indX,:]
     y[indX,y1] = 1
  w = []
  sD = D
  for i in range(N):
     w.append(np.random.uniform(-1.0, 1.0,(sD,K[i])))
     sD = K[i]
  w.append(np.random.uniform(-1.0, 1.0,(sD,10)))

  startB = 0
  for iter in xrange(mIter):
   while startB < X.shape[0]:
    endB = startB + batch
    if endB >= X.shape[0]:
     endB = X.shape[0]

    xB = X[startB:endB,:]
    yB = y[startB:endB,:]
   
    wX = []
    h = []
    xx = xB
    for i in range(N + 1):
      wX.append(np.dot(xx,w[i]))
      h.append(sig(wX[i]))
      xx = h[i]

    loss = []
    zz = []
    loss.append(yB - h[N])
    zz.append(loss[0] * sigDerivative(h[N]))
    for i in range(N):
     ll = N - i
     loss.append(zz[i].dot(w[ll].T))
     zz.append(loss[i + 1] * sigDerivative(h[ll - 1]))
   
   
    for i in range(N):
       ll = N - i
       w[ll] = w[ll] + lr * h[ll -1].T.dot(zz[i])
    w[0] = w[0] + lr * xB.T.dot(zz[N])

    startB = endB
  return (w)


def NNMLPTest(w):
   h = testX
   for i in range(len(w)):
      wX = np.dot(h,w[i])
      h = sig(wX)
   h2 = h
   yH = [np.argmax(h2[i]) for i in range(h2.shape[0])]
   cAr = [1 for i in range(len(yH)) if yH[i] == testY[i]]
   cNo = np.sum(cAr)
   acc = float(cNo) / float(len(testY))
   print "No of Matches = ", str(cNo), ", Accuracy = ",str(acc)

def activationB(actParm,a):
  h = []
  if actParm == 1:
    h = np.tanh(a)
  return h

def tanHDerivative(a):
    ai = np.tanh(a)
    ai = ai ** 2
    gi = np.zeros(ai.shape)
    for i in range(ai.shape[0]):
      gi[i] = 1.0 - ai[i]
    return gi


def NNTanH(K,lr,mIter,actParm):
  batch = 100
  D = X.shape[1]

  y = np.zeros((X.shape[0],10))
  for indX in range(X.shape[0]):
     y1 = Y[indX,:]
     y[indX,y1] = 1

  w1 = np.random.uniform(-1.0, 1.0,(D,K))
  w2 = np.random.uniform(-1.0, 1.0, (K,10))
  startB = 0
  for iter in xrange(mIter):
   while startB < X.shape[0]:
    endB = startB + batch
    if endB >= X.shape[0]:
     endB = X.shape[0]

    xB = X[startB:endB,:]
    yB = y[startB:endB,:]

    wX1 = np.dot(xB,w1)
    h1 = activationB(1,wX1)
    wX2 = np.dot(h1,w2)
    h2 = activationB(1,wX2)

    loss2 = yB - h2
    z2 = loss2 * tanHDerivative(wX2)
    w2_2 = h1.T.dot(z2)

    loss1 = z2.dot(w2.T)
    z1 = loss1 * tanHDerivative(wX1)
    w1_2 = xB.T.dot(z1)

    w2 = w2 + lr * w2_2
    w1 = w1 + lr * w1_2

    startB = endB
  return (w1,w2)



def NNTestTanh(w1,w2):
   wX1 = np.dot(testX,w1)
   h1 = activationB(1,wX1)
   wX2 = np.dot(h1,w2)
   h2 = activationB(1,wX2)
   yH = [np.argmax(h2[i]) for i in range(h2.shape[0])]
   cAr = [1 for i in range(len(yH)) if yH[i] == testY[i]]
   cNo = np.sum(cAr)
   acc = float(cNo) / float(len(testY))
   print "No of Matches = ", str(cNo), ", Accuracy = ",str(acc)



