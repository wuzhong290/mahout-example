# mahout-logistic-regression
Apache Mahout logistic regression demo. Code exlained here:
http://technobium.com/logistic-regression-using-apache-mahout/
http://blog.trifork.com/2014/02/04/an-introduction-to-mahouts-logistic-regression-sgd-classifier/
http://svn.apache.org/repos/asf/mahout/trunk/
Multinomial logistic regression

一、logistic-regression、
使用条件：仅仅支持二分类

1，根据数据训练生成model
mahout trainlogistic --input input/donut.csv --output ./model --target color --categories 2 --predictors x y shape color xx xy yy c a b --types numeric numeric numeric numeric numeric numeric numeric numeric numeric numeric --features 100 --passes 100 --rate 100
结果：

100
color ~
-7.262*Intercept Term + -28.320*a + -9.131*b + -68.306*c + -261.966*color + 17.644*shape + -20.696*x + -39.885*xx + -12.890*xy + -0.447*y + -24.766*yy
      Intercept Term -7.26219
                   a -28.31969
                   b -9.13138
                   c -68.30563
               color -261.96552
               shape 17.64429
                   x -20.69578
                  xx -39.88514
                  xy -12.88975
                   y -0.44683
                  yy -24.76592


2、model评价
mahout runlogistic --input input/donut-test.csv --model ./model --auc --confusion

AUC = 1.00
confusion: [[27.0, 0.0], [0.0, 13.0]]
entropy: [[-0.0, NaN], [-46.1, -0.0]]

AUC接近1.00表明：预测效果很好
AUC 和confusion表示分类准确率 AUC（readingData的正确率越接近1越好） confusion（识别率和误识率）

word
model age mileage

mahout trainlogistic --input input/inputData.csv --output ./model --target result --categories 2 --predictors model age mileage --types word numeric numeric --features 1000 --passes 10000 --rate 100

1000
result ~
491.912*Intercept Term + 2873.830*age + -0.297*mileage + 65.363*model=family + 356.646*model=medium + 484.422*model=small + 579.413*model=sport
      Intercept Term 491.91222
                 age 2873.83047
             mileage -0.29717
        model=family 65.36269
        model=medium 356.64575
         model=small 484.42207
         model=sport 579.41343

mahout runlogistic --input input/inputData-test.csv --model ./model --auc --confusion

AUC = 1.00
confusion: [[1.0, 0.0], [0.0, 1.0]]
entropy: [[-0.0, -Infinity], [-23.0, 0.0]]


http://blog.csdn.net/fansy1990/article/details/23858221

//mahout中的逻辑回归函数逻辑，其中：r为491.912*Intercept Term + 2873.830*age + -0.297*mileage + 65.363*model=family + 356.646*model=medium + 484.422*model=small + 579.413*model=sport
//        if (r < 0.0) {
//            double s = Math.exp(r);
//            return s / (1.0 + s);
//        } else {
//            double s = Math.exp(-r);
//            return 1.0 / (1.0 + s);
//        }

//small,1,2000,1  3256.1639999999998
//491.912*Intercept Term + 2873.830*age + -0.297*mileage + 65.363*model=family + 356.646*model=medium + 484.422*model=small + 579.413*model=sport
//Intercept Term 是个常量为1
double sum = 491.912*1 + 2873.830*1 + -0.297*2000 + 484.422;
System.out.println(sum);
//评分公式：double s = Math.exp(-r); 1.0 / (1.0 + s);  结果为1.0
double s = Math.exp(-sum);
System.out.println(1.0 / (1.0 + s));

//family,10,100000,0  -404.42500000000047
//491.912*Intercept Term + 2873.830*age + -0.297*mileage + 65.363*model=family + 356.646*model=medium + 484.422*model=small + 579.413*model=sport
//Intercept Term 是个常量为1
sum = 491.912*1 + 2873.830*10 + -0.297*100000 + 65.363;
System.out.println(sum);
//评分公式：double s = Math.exp(sum); s / (1.0 + s);  结果为2.293264542791767E-176 接近于零
s = Math.exp(sum);
System.out.println(s / (1.0 + s));



二、贝叶斯分类算法
mvn clean package assembly:single
https://chimpler.wordpress.com/2013/03/13/using-the-mahout-naive-bayes-classifier-to-automatically-classify-twitter-messages/

1、Training the model with Mahout
1.1、First we need to convert the training set to the hadoop sequence file format:
com.chimpler.example.bayes.TweetTSVToSeq input/bayes/data/tweets-train.tsv tweets-seq
1.2、upload this file to HDFS:
scp -r tweets-seq root@192.168.120.129:/home/wuzhong/
hadoop fs -put tweets-seq tweets-seq
1.3、run mahout to transform the training sets into vectors using tfidf weights(term frequency x document frequency):
mahout seq2sparse -i tweets-seq -o tweets-vectors
    It will generate the following files in HDFS in the directory tweets-vectors:
        df-count: sequence file with association word id => number of document containing this word
        dictionary.file-0: sequence file with association word => word id
        frequency.file-0: sequence file with association word id => word count
        tf-vectors: sequence file with the term frequency for each document
        tfidf-vectors: sequence file with association document id => tfidf weight for each word in the document
        tokenized-documents: sequence file with association document id => list of words
        wordcount: sequence file with association word => word count
1.4、Mahout splits the set into two sets: a training set and a testing set:
mahout split -i tweets-vectors/tfidf-vectors --trainingOutput train-vectors --testOutput test-vectors --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential
1.5、use the training set to train the classifier:
mahout trainnb -i train-vectors -el -li labelindex -o model -ow -c

2、Testing the model with Mahout
2.1、test that the classifier is working properly on the training set:
mahout testnb -i train-vectors -m model -l labelindex -ow -o tweets-testing -c
结果：
16/05/25 10:09:25 INFO test.TestNaiveBayesDriver: Complementary Results:
=======================================================
Summary
-------------------------------------------------------
Correctly Classified Instances          :        317       97.2393%
Incorrectly Classified Instances        :          9        2.7607%
Total Classified Instances              :        326

=======================================================
Confusion Matrix
-------------------------------------------------------
a       b       c       d       e       f       g       <--Classified as
63      0       0       0       0       0       0        |  63          a     = apparel
0       32      0       0       0       0       1        |  33          b     = art
0       0       35      0       0       0       0        |  35          c     = camera
1       0       0       37      0       0       0        |  38          d     = event
0       0       0       0       32      0       0        |  32          e     = health
0       3       0       0       0       29      1        |  33          f     = home
0       0       3       0       0       0       89       |  92          g     = tech

=======================================================
Statistics
-------------------------------------------------------
Kappa                                        0.926
Accuracy                                   97.2393%
Reliability                                84.8695%
Reliability (standard deviation)            0.3452

16/05/25 10:09:25 INFO driver.MahoutDriver: Program took 16388 ms (Minutes: 0.27313333333333334)


mahout testnb -i test-vectors -m model -l labelindex -ow -o tweets-testing -c
结果：
16/05/25 10:14:49 INFO test.TestNaiveBayesDriver: Complementary Results:
=======================================================
Summary
-------------------------------------------------------
Correctly Classified Instances          :        123        66.129%
Incorrectly Classified Instances        :         63        33.871%
Total Classified Instances              :        186

=======================================================
Confusion Matrix
-------------------------------------------------------
a       b       c       d       e       f       g       <--Classified as
23      4       0       0       2       4       3        |  36          a     = apparel
5       14      0       2       3       1       7        |  32          b     = art
0       0       22      1       1       0       1        |  25          c     = camera
0       4       0       14      4       0       1        |  23          d     = event
0       0       0       2       18      2       1        |  23          e     = health
1       1       2       2       0       15      3        |  24          f     = home
0       0       4       0       2       0       17       |  23          g     = tech

=======================================================
Statistics
-------------------------------------------------------
Kappa                                       0.5137
Accuracy                                    66.129%
Reliability                                58.8978%
Reliability (standard deviation)            0.2722

16/05/25 10:14:49 INFO driver.MahoutDriver: Program took 13173 ms (Minutes: 0.21955)

2.2、To use the classifier to classify new documents, we would need to copy several files from HDFS:
    model (matrix word id x label id)
    labelindex (mapping between a label and its id)
    dictionary.file-0 (mapping between a word and its id)
    df-count (document frequency: number of documents each word is appearing in)
$ hadoop fs -get labelindex labelindex
$ hadoop fs -get model model
$ hadoop fs -get tweets-vectors/dictionary.file-0 dictionary.file-0
$ hadoop fs -getmerge tweets-vectors/df-count df-count

scp root@192.168.120.129:/home/wuzhong/labelindex labelindex
scp root@192.168.120.129:/home/wuzhong/model model
scp root@192.168.120.129:/home/wuzhong/dictionary.file-0 dictionary.file-0
scp root@192.168.120.129:/home/wuzhong/df-count df-count

3、classify new tweets
com.chimpler.example.bayes.Classifier input/bayes/model/model input/bayes/model/labelindex input/bayes/model/dictionary.file-0 input/bayes/model/df-count input/bayes/data/tweets-to-classify.tsv

com.chimpler.example.bayes.TopCategoryWords input/bayes/model/model input/bayes/model/labelindex input/bayes/model/dictionary.file-0 input/bayes/model/df-count

com.chimpler.example.bayes.TweetTSVToTrainingSetSeq input/bayes/model/dictionary.file-0 input/bayes/model/df-count input/bayes/data/tweets-train.tsv  tweets-training-set