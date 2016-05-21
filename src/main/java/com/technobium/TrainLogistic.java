package com.technobium;


import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import com.google.common.io.Resources;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.classifier.sgd.LogisticModelParameters;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.RecordFactory;
import org.apache.mahout.math.RandomAccessSparseVector;

public final class TrainLogistic {
    private static String inputFile;
    private static String outputFile;
    private static LogisticModelParameters lmp;
    private static int passes;
    private static boolean scores;
    private static OnlineLogisticRegression model;

    private TrainLogistic() {
    }

    public static void main(String[] args) throws Exception {
        mainToOutput(args, new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }

    static void mainToOutput(String[] args, PrintWriter output) throws Exception {
        if(parseArgs(args)) {
            double logPEstimate = 0.0D;
            int samples = 0;
            CsvRecordFactory csv = lmp.getCsvRecordFactory();
            OnlineLogisticRegression lr = lmp.createRegression();

            double weight;
            for(int modelOutput = 0; modelOutput < passes; ++modelOutput) {
                output.printf(Locale.SIMPLIFIED_CHINESE,"训练次数为："+modelOutput);
                output.println();
                BufferedReader sep = open(inputFile);

                try {
                    csv.firstLine(sep.readLine());

                    for(String row = sep.readLine(); row != null; row = sep.readLine()) {
                        RandomAccessSparseVector column = new RandomAccessSparseVector(lmp.getNumFeatures());
                        int key = csv.processLine(row, column);
                        weight = lr.logLikelihood(key, column);
                        if(!Double.isInfinite(weight)) {
                            if(samples < 20) {
                                logPEstimate = ((double)samples * logPEstimate + weight) / (double)(samples + 1);
                            } else {
                                logPEstimate = 0.95D * logPEstimate + 0.05D * weight;
                            }

                            ++samples;
                        }

                        double p = lr.classifyScalar(column);
                        if(scores) {
                            output.printf(Locale.ENGLISH, "%10d %2d %10.2f %2.4f %10.4f %10.4f%n", new Object[]{Integer.valueOf(samples), Integer.valueOf(key), Double.valueOf(lr.currentLearningRate()), Double.valueOf(p), Double.valueOf(weight), Double.valueOf(logPEstimate)});
                        }

                        lr.train(key, column);
                    }
                } finally {
                    Closeables.close(sep, true);
                }
            }

            FileOutputStream var24 = new FileOutputStream(outputFile);

            try {
                lmp.saveTo(var24);
            } finally {
                Closeables.close(var24, false);
            }

            output.println(lmp.getNumFeatures());
            output.println(lmp.getTargetVariable() + " ~ ");
            String var25 = "";
            Iterator var26 = csv.getTraceDictionary().keySet().iterator();

            while(var26.hasNext()) {
                String var28 = (String)var26.next();
                double var31 = predictorWeight(lr, 0, csv, var28);
                if(var31 != 0.0D) {
                    output.printf(Locale.ENGLISH, "%s%.3f*%s", new Object[]{var25, Double.valueOf(var31), var28});
                    var25 = " + ";
                }
            }

            output.printf("%n", new Object[0]);
            model = lr;

            for(int var27 = 0; var27 < lr.getBeta().numRows(); ++var27) {
                Iterator var29 = csv.getTraceDictionary().keySet().iterator();

                while(var29.hasNext()) {
                    String var32 = (String)var29.next();
                    weight = predictorWeight(lr, var27, csv, var32);
                    if(weight != 0.0D) {
                        output.printf(Locale.ENGLISH, "%20s %.5f%n", new Object[]{var32, Double.valueOf(weight)});
                    }
                }

                for(int var30 = 0; var30 < lr.getBeta().numCols(); ++var30) {
                    output.printf(Locale.ENGLISH, "%15.9f ", new Object[]{Double.valueOf(lr.getBeta().get(var27, var30))});
                }

                output.println();
            }
        }

    }

    private static double predictorWeight(OnlineLogisticRegression lr, int row, RecordFactory csv, String predictor) {
        double weight = 0.0D;

        Integer column;
        for(Iterator i$ = ((Set)csv.getTraceDictionary().get(predictor)).iterator(); i$.hasNext(); weight += lr.getBeta().get(row, column.intValue())) {
            column = (Integer)i$.next();
        }

        return weight;
    }

    private static boolean parseArgs(String[] args) {
        DefaultOptionBuilder builder = new DefaultOptionBuilder();
        DefaultOption help = builder.withLongName("help").withDescription("print this list").create();
        DefaultOption quiet = builder.withLongName("quiet").withDescription("be extra quiet").create();
        DefaultOption scores = builder.withLongName("scores").withDescription("output score diagnostics during training").create();
        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        DefaultOption inputFile = builder.withLongName("input").withRequired(true).withArgument(argumentBuilder.withName("input").withMaximum(1).create()).withDescription("where to get training data").create();
        DefaultOption outputFile = builder.withLongName("output").withRequired(true).withArgument(argumentBuilder.withName("output").withMaximum(1).create()).withDescription("where to get training data").create();
        DefaultOption predictors = builder.withLongName("predictors").withRequired(true).withArgument(argumentBuilder.withName("p").create()).withDescription("a list of predictor variables").create();
        DefaultOption types = builder.withLongName("types").withRequired(true).withArgument(argumentBuilder.withName("t").create()).withDescription("a list of predictor variable types (numeric, word, or text)").create();
        DefaultOption target = builder.withLongName("target").withRequired(true).withArgument(argumentBuilder.withName("target").withMaximum(1).create()).withDescription("the name of the target variable").create();
        DefaultOption features = builder.withLongName("features").withArgument(argumentBuilder.withName("numFeatures").withDefault("1000").withMaximum(1).create()).withDescription("the number of internal hashed features to use").create();
        DefaultOption passes = builder.withLongName("passes").withArgument(argumentBuilder.withName("passes").withDefault("2").withMaximum(1).create()).withDescription("the number of times to pass over the input data").create();
        DefaultOption lambda = builder.withLongName("lambda").withArgument(argumentBuilder.withName("lambda").withDefault("1e-4").withMaximum(1).create()).withDescription("the amount of coefficient decay to use").create();
        DefaultOption rate = builder.withLongName("rate").withArgument(argumentBuilder.withName("learningRate").withDefault("1e-3").withMaximum(1).create()).withDescription("the learning rate").create();
        DefaultOption noBias = builder.withLongName("noBias").withDescription("don\'t include a bias term").create();
        DefaultOption targetCategories = builder.withLongName("categories").withRequired(true).withArgument(argumentBuilder.withName("number").withMaximum(1).create()).withDescription("the number of target categories to be considered").create();
        Group normalArgs = (new GroupBuilder()).withOption(help).withOption(quiet).withOption(scores).withOption(inputFile).withOption(outputFile).withOption(target).withOption(targetCategories).withOption(predictors).withOption(types).withOption(passes).withOption(lambda).withOption(rate).withOption(noBias).withOption(features).create();
        Parser parser = new Parser();
        parser.setHelpOption(help);
        parser.setHelpTrigger("--help");
        parser.setGroup(normalArgs);
        parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
        CommandLine cmdLine = parser.parseAndHelp(args);
        if(cmdLine == null) {
            return false;
        } else {
            TrainLogistic.inputFile = getStringArgument(cmdLine, inputFile);
            TrainLogistic.outputFile = getStringArgument(cmdLine, outputFile);
            ArrayList typeList = Lists.newArrayList();
            Iterator predictorList = cmdLine.getValues(types).iterator();

            while(predictorList.hasNext()) {
                Object i$ = predictorList.next();
                typeList.add(i$.toString());
            }

            ArrayList predictorList1 = Lists.newArrayList();
            Iterator i$1 = cmdLine.getValues(predictors).iterator();

            while(i$1.hasNext()) {
                Object x = i$1.next();
                predictorList1.add(x.toString());
            }

            lmp = new LogisticModelParameters();
            lmp.setTargetVariable(getStringArgument(cmdLine, target));
            lmp.setMaxTargetCategories(getIntegerArgument(cmdLine, targetCategories));
            lmp.setNumFeatures(getIntegerArgument(cmdLine, features));
            lmp.setUseBias(!getBooleanArgument(cmdLine, noBias));
            lmp.setTypeMap(predictorList1, typeList);
            lmp.setLambda(getDoubleArgument(cmdLine, lambda));
            lmp.setLearningRate(getDoubleArgument(cmdLine, rate));
            TrainLogistic.scores = getBooleanArgument(cmdLine, scores);
            TrainLogistic.passes = getIntegerArgument(cmdLine, passes);
            return true;
        }
    }

    private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
        return (String)cmdLine.getValue(inputFile);
    }

    private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
        return cmdLine.hasOption(option);
    }

    private static int getIntegerArgument(CommandLine cmdLine, Option features) {
        return Integer.parseInt((String)cmdLine.getValue(features));
    }

    private static double getDoubleArgument(CommandLine cmdLine, Option op) {
        return Double.parseDouble((String)cmdLine.getValue(op));
    }

    public static OnlineLogisticRegression getModel() {
        return model;
    }

    public static LogisticModelParameters getParameters() {
        return lmp;
    }

    static BufferedReader open(String inputFile) throws IOException {
        Object in;
        try {
            in = Resources.getResource(inputFile).openStream();
        } catch (IllegalArgumentException var3) {
            in = new FileInputStream(new File(inputFile));
        }

        return new BufferedReader(new InputStreamReader((InputStream)in, Charsets.UTF_8));
    }
}

