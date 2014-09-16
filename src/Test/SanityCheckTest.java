package Test;

import Accessories.MSTReader;
import Classifer.AveragedPerceptron;
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
        System.out.println("Hello world!");
        String trainPath=   "/Users/msr/Projects/mstparser_0.2/MSTParser/data/train.lab";
        String devPath="/Users/msr/Projects/mstparser_0.2/MSTParser/data/test.lab";
        String  modelPath="/Users/msr/Projects/mstparser_0.2/MSTParser/data/model";
        if(args.length>=3){
            trainPath=args[0];
            devPath=args[1];
            modelPath=args[2];
        }

        AveragedPerceptron perceptron=new AveragedPerceptron(44);

        ArrayList<Sentence> trainData= MSTReader.readSentences(trainPath);
        ArrayList<Sentence> devData=MSTReader.readSentences(devPath);
        ArrayList<String> possibleLabels=new ArrayList<String>();
        possibleLabels.add("");

        PartialTreeTrainer.train(trainData,devData,possibleLabels,perceptron,modelPath,30);
    }
}
