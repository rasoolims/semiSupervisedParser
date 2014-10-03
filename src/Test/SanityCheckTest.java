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

        boolean weighted = false;
        if (args.length >= 6) {
            trainPath = args[0];
            devPath = args[1];
            modelPath = args[2];
            weighted = Boolean.parseBoolean(args[3]);
            useDynamTrain = Boolean.parseBoolean(args[4]);
            labeled = Boolean.parseBoolean(args[5]);
            secondOrder=Boolean.parseBoolean(args[6]);
        }

        System.out.println("weighted:\t" + weighted);
        System.out.println("second_order:\t" + secondOrder);

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
        PartialTreeTrainer.train(trainData, devData, possibleLabels, perceptron, modelPath, 30, useDynamTrain, modelPath + ".out");
        else
             PartialTreeTrainer.train2ndOrder(trainData, devData, possibleLabels, perceptron, modelPath, 30, modelPath + ".out");

    }
}
