import util
  
option = 4

util.getDS()

if option == 1:
  util.splitTrainSequential(0.5)
  util.testSplitTrainSequential(2400,800)
  util.splitStratifiedTrainSequential(0.5)
  util.splitStratifiedTrainRandom(0.6)
  util.countSplit()
  util.histogram('sequential-random.png')


if option == 2:
  util.splitStratifiedTrainRandom(0.8)
  util.getXY(1)
  (W,b) = util.MLPtrain(1000,2)
  util.MLPtest(W,b)

if option == 3:
  util.splitStratifiedTrainRandom(0.8)
  util.getXY(1)
  lrParm = 1 # 1 for fixed learning rate
	     # 2 for AdaGrad learning rate
  (w1,w2) = util.NN(30,0.1,60000,1,lrParm)
  util.NNTest(w1,w2)

if option == 4:
  util.splitStratifiedTrainRandom(0.8)
  util.getXY(1)
  layers = [30,20]
  n = len(layers)
  w = util.NNMLP(n,layers,0.1,60000,1)
  util.NNMLPTest(w)

if option == 5:
  util.splitStratifiedTrainRandom(0.8)
  util.getXY(1)
  (w1,w2) = util.NNTanH(30,0.01,6000000,1)
  util.NNTestTanh(w1,w2)

if option == 6:
  util.splitStratifiedTrainRandom(0.8)
  util.getXY(2)
  print "Perceptron with fixed learning rate"
  (W,b) = util.MLPtrain(100,1)
  util.MLPtest(W,b)

  print "Neural Network with 1 hidden layer, 20 nodes, fixed learning rate"
  lrParm = 1 # 1 for fixed learning rate
             # 2 for AdaGrad learning rate
  (w1,w2) = util.NN(30,0.1,60000,1,lrParm)
  util.NNTest(w1,w2)
