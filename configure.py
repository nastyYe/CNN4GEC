#!/usr/bin/python
#-*- coding:utf-8 -*-

# 设置工程的目录
phome = '/home/xqjin/'
vectorL = "50"
vectorT = "T"  # W: wiki  G:google  T:turian  E:error driven  U: unifrom
crossvalidate = ""  # "_cv0"  ""    判断使用那份数据，作为训练和测试

# 设置语料库的路径:训练语料,测试语料以及检验语料
pCorpusTrain = phome + 'Graduate/Corpus/NUCLE/train/'
pCorpusTest  = phome + 'Graduate/Corpus/NUCLE/test/'
pCorpusCV = phome + 'Graduate/Corpus/NUCLE/cvdata/'

pCorpusPickle = phome + 'Graduate/Corpus/NUCLE/pickle/'

pTokenFilePrep = phome + 'Graduate/Corpus/Prep/tokenFeature/'
pVectorFilePrep = phome + 'Graduate/Corpus/Prep/vectorFeature/'

pTokenFileArt = phome + 'Graduate/Corpus/ArtOrDet/tokenFeature/'
pVectorFileArt = phome + 'Graduate/Corpus/ArtOrDet/vectorFeature/'


pTokenFileNn = phome + 'Graduate/Corpus/Nn/tokenFeature/'
pVectorFileNn = phome + 'Graduate/Corpus/Nn/vectorFeature/'

pCorpusWord2vec = phome + "Graduate/Corpus/word2vec/" + "vector%s/" %vectorL

pOutputPrep = phome + "Graduate/Output/Prep/"
pOutputArt = phome + "Graduate/Output/ArtOrDet/"
pOutputNn = phome + "Graduate/Output/Nn/"

# train *.conll 文件的位置,已经处理pickle结果存放的位置;
fCorpusTrainConll = pCorpusTrain + 'conll13st-preprocessed.conll'
fCorpusPickleTrainConll = pCorpusPickle + 'train%s.conll' %crossvalidate
fCorpusPickleTrainSentTree = pCorpusPickle + 'train%s.senttree'  %crossvalidate
fCorpusPickleTrainSentence = pCorpusPickle + 'train%s.sentence'  %crossvalidate

# test *.conll 文件的位置,已经处理pickle结果存放的位置;
fCorpusTestConll = pCorpusTest + 'official-preprocessed.conll'
fCorpusPickleTestConll = pCorpusPickle + 'test%s.conll' %crossvalidate
fCorpusPickleTestSentTree = pCorpusPickle + 'test%s.senttree' %crossvalidate
fCorpusPickleTestSentence = pCorpusPickle + 'test%s.sentence' %crossvalidate

# modify file of the prep!
fCorpusTestM2 = pCorpusTest + "official-preprocessed.m2"
fCorpusTestPrepM2 = pCorpusTest + "Perp.m2"
fCorpusTestArtOrDetM2 = pCorpusTest + "ArtOrDet.m2"
fCorpusTestNnM2 = pCorpusTest + "Nn.m2"

# for 
fCorpusTestCVM2 = pCorpusCV + "test%s.m2"  %crossvalidate
fCorpusTestCVPrepM2 = pCorpusCV + "PrepCV%s.m2"  %crossvalidate
fCorpusTestCVArtOrDetM2 = pCorpusCV + "ArtOrDetCV%s.m2"  %crossvalidate
fCorpusTestCVNnM2 = pCorpusCV + "Nn%s.m2"  %crossvalidate



# train *.ann position
fCorpusTrainAnn = pCorpusTrain + 'conll13st-preprocessed.conll.ann'
fCorpusPickleTrainAnn = pCorpusPickle + 'train%s.ann'  %crossvalidate

# test *.ann position
fCorpusTestAnn = pCorpusTest + 'official-preprocessed.conll.ann'
fCorpusPickleTestAnn = pCorpusPickle + 'test%s.ann'  %crossvalidate

#the word2vec's path 
fword2vecWI  = pCorpusWord2vec + "index/word_index.dict"
fword2vecUWI = pCorpusWord2vec + "uindex/word_index.dict"
fword2vecVec = pCorpusWord2vec + "vector/word2vec%s.dict"
fword2vecUVec = pCorpusWord2vec + "uvector/word2vec%s.dict"


#######  The upper is the same  ########   The upper is the same   ########

#######  The upper is the same  ########   The upper is the same   ########

## token file
fTrainTokenPrep = pTokenFilePrep + "trainTokenFeature%s.prep"  %crossvalidate
fTestTokenPrep  = pTokenFilePrep + "testTokenFeature%s.prep"  %crossvalidate

fTrainTokenArt = pTokenFileArt + "trainTokenFeature%s.art"  %crossvalidate
fTestTokenArt  = pTokenFileArt + "testTokenFeature%s.art"  %crossvalidate

fTrainTokenNn = pTokenFileNn + "trainTokenFeature%s.nn"  %crossvalidate
fTestTokenNn  = pTokenFileNn + "testTokenFeature%s.nn"  %crossvalidate

## IW  Vec and train test vector used in the NUCLE corpus!

fTrainTestIWPrep = pVectorFilePrep + "temp/traintestIW%s.dict" %(vectorL+crossvalidate)
fTrainTestVecPrep = pVectorFilePrep + "temp/traintestVec%s.dict" %(vectorL+crossvalidate)

fTrainTestIWArt = pVectorFileArt + "temp/traintestIW%s.dict" %(vectorL+crossvalidate)
fTrainTestVecArt = pVectorFileArt + "temp/traintestVec%s.dict" %(vectorL+crossvalidate)

fTrainTestIWNn = pVectorFileNn + "temp/traintestIW%s.dict" %(vectorL+crossvalidate)
fTrainTestVecNn = pVectorFileNn + "temp/traintestVec%s.dict" %(vectorL+crossvalidate)


## train & test vector for DNN

fTrainVecPrep = pVectorFilePrep + "train.vec"
fTrainUVecPrep = pVectorFilePrep + "train.uvec"

fTrainVecArt = pVectorFileArt + "train.vec"
fTrainUVecArt = pVectorFileArt + "train.uvec"



fSubTrainVecPrep = pVectorFilePrep + "subtrain.vec"
fSubTrainUVecPrep = pVectorFilePrep + "subtrain.uvec"

fSubTrainVecArt = pVectorFileArt + "subtrain.vec"
fSubTrainUVecArt = pVectorFileArt + "subtrain.uvec"



fValidateVecPrep = pVectorFilePrep + "validate.vec"
fValidateUVecPrep = pVectorFilePrep + "validate.uvec"

fValidateVecArt = pVectorFileArt + "validate.vec"
fValidateUVecArt = pVectorFileArt + "validate.uvec"



fTestVecPrep = pVectorFilePrep + "test.vec"
fTestUVecPrep = pVectorFilePrep + "test.uvec"

fTestVecArt = pVectorFileArt + "test.vec"
fTestUVecArt = pVectorFileArt + "test.uvec"

###########

fDNNModelPrep = pOutputPrep + "DNN/model/DNN-"
fDNNResultPrep =pOutputPrep + "DNN/Result/DNN.result"
fDNNCorrectResPrep =pOutputPrep + "DNN/Result/DNNCorrect.result"

fDNNModelArt = pOutputArt + "DNN/model/DNN-"
fDNNResultArt =pOutputArt + "DNN/Result/DNN.result"
fDNNCorrectResArt =pOutputArt + "DNN/Result/DNNCorrect.result"


fCNNModelPrep = pOutputPrep + "CNN/model/CNN%s-"  %(vectorT+vectorL+crossvalidate)
fCNNResultPrep = pOutputPrep + "CNN/Result/CNN%s.result"  %(vectorT+vectorL+crossvalidate)
fCNNCorrectResPrep = pOutputPrep + "CNN/Result/CNNCorrect%s.result"  %(vectorT+vectorL+crossvalidate)

fCNNModelArt = pOutputArt + "CNN/model/CNN%s-"  %(vectorT+vectorL+crossvalidate)
fCNNResultArt = pOutputArt + "CNN/Result/CNN%s.result"  %(vectorT+vectorL+crossvalidate)
fCNNCorrectResArt = pOutputArt + "CNN/Result/CNNCorrect%s.result"  %(vectorT+vectorL+crossvalidate)


fCNNModelNn = pOutputNn + "CNN/model/CNN%s-"  %(vectorT+vectorL+crossvalidate)
fCNNResultNn = pOutputNn + "CNN/Result/CNN%s.result"  %(vectorT+vectorL+crossvalidate)
fCNNCorrectResNn = pOutputNn + "CNN/Result/CNNCorrect%s.result"  %(vectorT+vectorL+crossvalidate)


fLMDataNn = pOutputNn + "LM/lm.data"
fLMModleNn = pOutputNn + "LM/lm.model"
fLMLinggleDataNn = pOutputNn + "LM/lm.linggle"

fLMDataArt = pOutputArt + "LM/lm.data"
fLMModleArt = pOutputArt + "LM/lm%s.model"
fLMLinggleDataArt = pOutputArt + "LM/lm.linggle"


"""
PCorpusWord2vec = phome + "Graduate/Corpus/word2vec/"
# 设置程序交叉验证使用相应的训练和测试数据，未经处理过的
pCorpusCV = phome + 'Graduate/Corpus/CV/'
# 设置程序处理语料的结果
pCorpusText = phome + 'Graduate/Corpus/text/'
pOutputResult = phome + 'Graduate/Output/result/'
pOutputModel = phome + 'Graduate/Output/model/'


pCVTokenFeature = phome + 'Graduate/Output/CVFeature/token/'
pCVNumFeature = phome + 'Graduate/Output/CVFeature/num/'

pTokenFeature = phome + 'Graduate/Output/Feature/token/'
pNumFeature = phome + 'Graduate/Output/Feature/num/'


# 保存提取出的特征，特征用token的形式表示，没有经过编码
pOutputTrainToken = pTokenFeature + 'train/'
pOutputTestToken = pTokenFeature + 'test/'

pOutputTrainNum = pNumFeature + 'train/'
pOutputTestNum = pNumFeature + 'test/'

# save the token and token's index file 
fCorpusPickleWordDictSet = pCorpusPickle + "word.set"
fCorpusPickleWordIndexDict = pCorpusPickle + "wordindex"

# word2vec 聚类结果文件
fCorpusTextClassSorted = pCorpusText + 'classes.sorted.txt'

# 将word2vec 聚类结果处理成： word：cluster_label 的形式的文件所存放的位置；
fCorpusPickleWord2Clu = pCorpusPickle + 'word2clu'

fCorpusNumInfo = pCorpusPickle + "NumInfo"

fOutputResultCorrect = pOutputResult + 'train_res'

fOutputModelArt = pOutputModel + 'train.model'

fCorpusTestM2 = pCorpusTest + "official-preprocessed.m2"
fOutputResultDet = pOutputResult + 'test.det'


pCorpusText = "/home/xqjin/Code/CodeBlocks/NN/model/"

pCorpusTextModel = pCorpusText + "dbn.model"

pCorpusTextParaA = pCorpusText + "modelA"
pCorpusTextParaB = pCorpusText + "modelB"
pCorpusTextParaC = pCorpusText + "modelC"
pCorpusTextParaD = pCorpusText + "modelD"
pCorpusTextParaP = pCorpusText + "modelP"

pCorpusTextParaVBE = pCorpusText + "modelVBE"

fCorpusWord2vecAcc = PCorpusWord2vec + "acc_vec"

#Can del del  del del del del del del  #Can del del  del del del del del del 

pOutputTrainNumWord2vec = phome + 'Graduate/Output/train/num/word2vec/'
pOutputTestNumWord2vec = phome + 'Graduate/Output/test/num/word2vec/'

# 存放编码后的文件的位置；
fOutputTrainNum = pOutputTrainNum + "train"
fOutputTestNum = pOutputTestNum + "test"

fOutputTrainNumWord2vec = pOutputTrainNumWord2vec + "train"
fOutputTestNumWord2vec = pOutputTestNumWord2vec + "test"

fCorpusWord2vecWordIndex = PCorpusWord2vec + "word_index.dict"
fCorpusWord2vecWordVec = PCorpusWord2vec + "word2vec"
"""
