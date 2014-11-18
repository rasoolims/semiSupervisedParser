package Test;

import Accessories.MSTReader;
import Classifier.AdaGrad;
import Classifier.AveragedPerceptron;
import Classifier.OnlineClassifier;
import Structures.Sentence;
import Trainer.PartialTreeTrainer;

import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * User: Mohammad Sadegh Rasooli
 * Date: 9/16/14
 * Time: 11:12 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class SanityCheckTest {
    public static void main(String[] args) throws Exception {
        String trainPath = "/Users/msr/Downloads/classifer.part.mst";
        String devPath = "/Users/msr/Downloads/dev.auto.tag.mst";
        String modelPath = "/Users/msr/Projects/mstparser_0.2/MSTParser/data/model";
        boolean useDynamTrain = true;
        boolean labeled = false;
        boolean secondOrder = true;
        boolean useHandCraftedRules = false;
        boolean trainPartial = false;
        int constrainedIterNum = -1;
        double contrainMinumumRatioDeps = 1.0;
        boolean iterativeConstraint = false;
        int iterativeConstraintPeriod = 3;
        boolean alwaysPartial = false;
        String classifierType = "perceptron";
        double learningRate = 1;
        double ridge = 0.1;
        boolean softConstrained=false;
        boolean secondOrderPartial=true;

        if (args.length > 3) {
            trainPath = args[0];
            devPath = args[1];
            modelPath = args[2];
            secondOrder = Boolean.parseBoolean(args[3]);
            if (args.length > 4)
                useHandCraftedRules = Boolean.parseBoolean(args[4]);
            if (args.length > 5)
                trainPartial = Boolean.parseBoolean(args[5]);
            if (args.length > 6)
                constrainedIterNum = Integer.parseInt(args[6]);
            if (args.length > 7)
                contrainMinumumRatioDeps = Double.parseDouble(args[7]);
            if (args.length > 8)
                iterativeConstraint = Boolean.parseBoolean(args[8]);
            if (args.length > 9)
                iterativeConstraintPeriod = Integer.parseInt(args[9]);
            if (args.length > 10)
            alwaysPartial = Boolean.parseBoolean(args[10]);
            if (args.length > 11)
                softConstrained = Boolean.parseBoolean(args[11]);
            if (args.length > 12)
                secondOrderPartial = Boolean.parseBoolean(args[12]);
            if (args.length > 13)
                classifierType = args[13];
            if (args.length > 14)
                learningRate = Double.parseDouble(args[14]);
            if (args.length > 15)
                ridge = Double.parseDouble(args[15]);
        } else {
            System.out.println("arguments: [train_path(mst_file)] [dev_path(mst_file)] [model_output_path]  [is_2nd_order(bool)]" +
                    " [use_linguistic_heuristics(bool)] [train_2nd_order_on_partial_trees(bool)] [constrainedIterNum] [contrainMinumumRatioDeps]" +
                    " [iterativeConstraint(bool)] [iterativeConstraintPeriod] [alwaysPartial(bool)] [soft_constrained(bool)] [secondOrderPartial(bool)] [classifier_type(percpetron/adagrad)] [adagrad_learning_rate] [ada_grad_ridge]");
        }

        System.out.println("dyn_train:\t" + useDynamTrain);
        System.out.println("second_order:\t" + secondOrder);
        System.out.println("use_ling_tricks:\t" + useHandCraftedRules);
        System.out.println("train_partial:\t" + trainPartial);
        System.out.println("constraint_iter:\t" + constrainedIterNum);
        System.out.println("contraint_prop:\t" + contrainMinumumRatioDeps);
        System.out.println("iterative Constraint:\t" + iterativeConstraint);
        System.out.println("iterative Constraint Period:\t" + iterativeConstraintPeriod);
        System.out.println("always Partial:\t" + alwaysPartial);
        System.out.println("soft constrained:\t" + softConstrained);
        System.out.println("partial second order:\t" + secondOrderPartial);
        System.out.println("classifier type:\t" + classifierType);
        System.out.println("learning rate:\t" + learningRate);
        System.out.println("ridge:\t" + ridge);

        OnlineClassifier onlineClassifier;
        if (classifierType.equals("adagrad"))
            onlineClassifier = new AdaGrad(learningRate, ridge);
        else

            onlineClassifier = new AveragedPerceptron();
        //        if(!secondOrder)
        //           perceptron=labeled?new AveragedPerceptron(64): new AveragedPerceptron(44);

        ArrayList<Sentence> devData = MSTReader.readSentences(devPath, true);
        ArrayList<String> possibleLabels = new ArrayList<String>();


        if (!secondOrder) {
            ArrayList<Sentence> trainData = MSTReader.readSentences(trainPath, false);
            if (labeled) {
                for (Sentence sentence : trainData) {
                    for (int i = 1; i < sentence.length(); i++) {
                        if (sentence.hasHead(i)) {
                            String label = sentence.label(i);
                            if (!label.equals("") && !possibleLabels.contains(label))
                                possibleLabels.add(label);
                        }
                    }
                }
            }
            if (possibleLabels.size() == 0)
                possibleLabels.add("");
            System.err.println("labeled: " + labeled + " with " + possibleLabels.size() + " possibilities");
            PartialTreeTrainer.train(trainData, devData, possibleLabels, onlineClassifier, modelPath, 30, useDynamTrain, modelPath + ".out", useHandCraftedRules,false);
        } else {
            if (possibleLabels.size() == 0)
                possibleLabels.add("");
            System.err.println("labeled: " + labeled + " with " + possibleLabels.size() + " possibilities");
            PartialTreeTrainer.train2ndOrder(trainPath, devData, possibleLabels, onlineClassifier, modelPath, 30, modelPath + ".out", useHandCraftedRules, trainPartial, constrainedIterNum, contrainMinumumRatioDeps, iterativeConstraint, iterativeConstraintPeriod, alwaysPartial,softConstrained,secondOrderPartial);
        }
    }
}
