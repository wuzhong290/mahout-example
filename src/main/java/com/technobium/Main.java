package com.technobium;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.io.Closer;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import java.io.*;

/**
 * Created by wuzhong on 2016/5/21.
 */
public class Main {

    public static void main(String[] args) {
        //使用训练程序生成的OnlineLogisticRegression进行评分
        classifyByOldMLR();
        //使用新建的OnlineLogisticRegression进行评分
        classifyBynewMLR();
    }

    private static void classifyByOldMLR(){
        Closer closer = Closer.create();

        OnlineLogisticRegression read;

        try {
            FileInputStream byteArrayInputStream = closer.register(new FileInputStream(new File("model.txt")));
            DataInputStream dataInputStream = closer.register(new DataInputStream(byteArrayInputStream));
            read = closer.register(PolymorphicWritable.read(dataInputStream, OnlineLogisticRegression.class));

            Vector v = new DenseVector(5);
            Splitter onComma = Splitter.on(",");
            v.set(0, 1);
            int i = 1;
            Iterable<String> values = onComma.split("6.0,2.7,5.1,1.6,versicolor");
            for (String value : Iterables.limit(values, 4)) {
                v.set(i++, Double.parseDouble(value));
            }

            Vector vt = read.classifyFull(v);

            Dictionary dict = new Dictionary();
            //保证和反序列化获取的OnlineLogisticRegression的target一致
            dict.intern("setosa");
            dict.intern("versicolor");
            dict.intern("virginica");
            int t = dict.intern(Iterables.get(values, 4));
            int r = vt.maxValueIndex();
            boolean flag =  r == t;
            System.out.println("");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                closer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void classifyBynewMLR(){
        OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 5, new L2(1));
        Vector v = new DenseVector(5);
        Splitter onComma = Splitter.on(",");
        v.set(0, 1);
        int i = 1;
        Iterable<String> values = onComma.split("6.0,2.7,5.1,1.6,versicolor");
        for (String value : Iterables.limit(values, 4)) {
            v.set(i++, Double.parseDouble(value));
        }

        Vector vt = lr.classifyFull(v);

        Dictionary dict = new Dictionary();
        //保证和反序列化获取的OnlineLogisticRegression的target一致
        dict.intern("setosa");
        dict.intern("versicolor");
        dict.intern("virginica");
        int t = dict.intern(Iterables.get(values, 4));
        int r = vt.maxValueIndex();
        boolean flag =  r == t;
        System.out.println("");
    }
}
