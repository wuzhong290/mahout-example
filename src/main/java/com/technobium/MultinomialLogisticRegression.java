package com.technobium;/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.Closer;
import com.google.common.io.Resources;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Random;

//多分类的处理
//训练后的OnlineLogisticRegression序列化到文件中，在使用的时候通过反序列化获取OnlineLogisticRegression
//Main.java是反序列化获取OnlineLogisticRegression，对未知数据进行评分。
public final class MultinomialLogisticRegression {

  private static final Logger logger = LoggerFactory.getLogger(MultinomialLogisticRegression.class);

  public static void main(String[] args) throws Exception{
    // this test trains a 3-way classifier on the famous Iris dataset.
    // a similar exercise can be accomplished in R using this code:
    //    library(nnet)
    //    correct = rep(0,100)
    //    for (j in 1:100) {
    //      i = order(runif(150))
    //      train = iris[i[1:100],]
    //      test = iris[i[101:150],]
    //      m = multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, train)
    //      correct[j] = mean(predict(m, newdata=test) == test$Species)
    //    }
    //    hist(correct)
    //
    // Note that depending on the training/test split, performance can be better or worse.
    // There is about a 5% chance of getting accuracy < 90% and about 20% chance of getting accuracy
    // of 100%
    //
    // This test uses a deterministic split that is neither outstandingly good nor bad


    RandomUtils.useTestSeed();
    Splitter onComma = Splitter.on(",");

    // read the data
    List<String> raw = Resources.readLines(Resources.getResource("iris.csv"), Charsets.UTF_8);

    // holds features
    List<Vector> data = Lists.newArrayList();

    // holds target variable
    List<Integer> target = Lists.newArrayList();

    // for decoding target values
    Dictionary dict = new Dictionary();

    // for permuting data later
    List<Integer> order = Lists.newArrayList();

    for (String line : raw.subList(1, raw.size())) {
      // order gets a list of indexes
      order.add(order.size());

      // parse the predictor variables
      Vector v = new DenseVector(5);
      v.set(0, 1);
      int i = 1;
      Iterable<String> values = onComma.split(line);
      for (String value : Iterables.limit(values, 4)) {
        v.set(i++, Double.parseDouble(value));
      }
      data.add(v);

      // and the target
      target.add(dict.intern(Iterables.get(values, 4)));
    }

    // randomize the order ... original data has each species all together
    // note that this randomization is deterministic
    Random random = RandomUtils.getRandom();
    Collections.shuffle(order, random);

    // select training and test data
    List<Integer> train = order.subList(0, 100);
    List<Integer> test = order.subList(100, 150);
    logger.warn("Training set = {}", train);
    logger.warn("Test set = {}", test);

    // now train many times and collect information on accuracy each time
    int[] correct = new int[test.size() + 1];
    for (int run = 0; run < 200; run++) {
      OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 5, new L2(1));
      // 30 training passes should converge to > 95% accuracy nearly always but never to 100%
      for (int pass = 0; pass < 30; pass++) {
        Collections.shuffle(train, random);
        for (int k : train) {
          lr.train(target.get(k), data.get(k));
        }
      }

      // check the accuracy on held out data
      int x = 0;
      int[] count = new int[3];
      for (Integer k : test) {
        Vector vt = lr.classifyFull(data.get(k));
        int r = vt.maxValueIndex();
        count[r]++;
        x += r == target.get(k) ? 1 : 0;
      }
      correct[x]++;

      if(run==199){

        Vector v = new DenseVector(5);
        v.set(0, 1);
        int i = 1;
        Iterable<String> values = onComma.split("6.0,2.7,5.1,1.6,versicolor");
        for (String value : Iterables.limit(values, 4)) {
          v.set(i++, Double.parseDouble(value));
        }

        Vector vt = lr.classifyFull(v);

        int r = vt.maxValueIndex();
        boolean flag =  r == dict.intern(Iterables.get(values, 4));

        lr.close();

        Closer closer = Closer.create();

        try {
          FileOutputStream byteArrayOutputStream = closer.register(new FileOutputStream(new File("model.txt")));
          DataOutputStream dataOutputStream = closer.register(new DataOutputStream(byteArrayOutputStream));
          PolymorphicWritable.write(dataOutputStream, lr);
        } finally {
          closer.close();
        }
      }
    }

    // verify we never saw worse than 95% correct,
    for (int i = 0; i < Math.floor(0.95 * test.size()); i++) {
      System.out.println(String.format("%d trials had unacceptable accuracy of only %.0f%%: ", correct[i], 100.0 * i / test.size()));
    }
    // nor perfect
    System.out.println(String.format("%d trials had unrealistic accuracy of 100%%", correct[test.size() - 1]));
  }


}