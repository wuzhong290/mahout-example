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

