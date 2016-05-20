package com.technobium;

/**
 * Created by wuzhong on 2016/5/19.
 */
public class Test {

    public static void main(String[] args) {
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
    }
}
