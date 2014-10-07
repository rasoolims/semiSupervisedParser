package Test;

import Accessories.MSTReader;
import Classifier.AveragedPerceptron;
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
        String trainPath = "/Users/msr/Desktop/full.train.mst";
        String devPath = "/Users/msr/Desktop/full.dev.mst";
        String modelPath = "/Users/msr/Projects/mstparser_0.2/MSTParser/data/model";
        boolean useDynamTrain = true;
        boolean labeled = false;
        boolean secondOrder=false;
        boolean useHandCraftedRules=false;
        boolean trainPartial=false;
        int constrainedIterNum=-1;
        double contrainMinumumRatioDeps=1.0;

        boolean weighted = false;
        if (args.length > 6) {
            trainPath = args[0];
            devPath = args[1];
            modelPath = args[2];
            weighted = Boolean.parseBoolean(args[3]);
            useDynamTrain = Boolean.parseBoolean(args[4]);
            labeled = Boolean.parseBoolean(args[5]);
            secondOrder=Boolean.parseBoolean(args[6]);
            if(args.length>7)
                useHandCraftedRules     =Boolean.parseBoolean(args[7]);
            if(args.length>8)
                trainPartial     =Boolean.parseBoolean(args[8]);
            if(args.length>9)
                constrainedIterNum     =Integer.parseInt(args[9]);
            if(args.length>10)
                contrainMinumumRatioDeps     =Double.parseDouble(args[10]);
        } else{
            System.out.println("arguments: [train_path(mst_file)] [dev_path(mst_file)] [model_output_path] [is_weighted(bool)] [use_dynamic_programming_train(bool)] [is_labeled(temporary not wordking)] [is_2nd_order(bool)] [use_linguistic_heuristics(bool)] [train_2nd_order_on_partial_trees(bool)]");
        }

        System.out.println("dyn_train:\t" + useDynamTrain);
        System.out.println("weighted:\t" + weighted);
        System.out.println("second_order:\t" + secondOrder);
        System.out.println("use_ling_tricks:\t" + useHandCraftedRules);
        System.out.println("train_partial:\t" + trainPartial);
        System.out.println("constraint_iter:\t" + constrainedIterNum);
        System.out.println("contraint_prop:\t" + contrainMinumumRatioDeps);

        AveragedPerceptron perceptron =   new AveragedPerceptron(1);
        //        if(!secondOrder)
         //           perceptron=labeled?new AveragedPerceptron(64): new AveragedPerceptron(44);

        ArrayList<Sentence> trainData = MSTReader.readSentences(trainPath, weighted);
        ArrayList<Sentence> devData = MSTReader.readSentences(devPath, false);
        ArrayList<String> possibleLabels = new ArrayList<String>();
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

        System.err.println("labeled: "+labeled+" with "+possibleLabels.size()+" possibilities");

         if(!secondOrder)
        PartialTreeTrainer.train(trainData, devData, possibleLabels, perceptron, modelPath, 30, useDynamTrain, modelPath + ".out",useHandCraftedRules);
        else
             PartialTreeTrainer.train2ndOrder(trainData, devData, possibleLabels, perceptron, modelPath, 30, modelPath + ".out",useHandCraftedRules,trainPartial,constrainedIterNum,contrainMinumumRatioDeps);

    }
}
