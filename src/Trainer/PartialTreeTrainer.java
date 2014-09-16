package Trainer;

import Classifer.AveragedPerceptron;
import Decoder.FeatureExtractor;
import Structures.Sentence;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 11:28 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class PartialTreeTrainer {
    public static void train(ArrayList<Sentence> trainSentences, ArrayList<Sentence> devSentences, ArrayList<String> possibleLabels,
                             AveragedPerceptron perceptron, String modelPath, int maxIter) throws Exception {

        int dimension = perceptron.dimension();

        for (int iter = 0; iter < maxIter; iter++) {
            int numDep=0;
            double correct=0;
            System.out.println("*********************************************************");
            System.out.println("iteration: " + iter);
            int senCount=0;
            for (Sentence sentence : trainSentences) {
                senCount++;
                if(senCount%100==0){
                    System.out.print(senCount+"...");
                }
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch)) {
                        numDep++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);

                        int argmax = 0;
                        String bestLabel = "";
                        double max = Double.NEGATIVE_INFINITY;

                        for (String label : possibleLabels) {
                            for (int h = 0; h < sentence.length(); h++) {
                                double score = perceptron.score(FeatureExtractor.extractFeatures(sentence, h, ch, label, dimension), false);
                                if (score > max) {
                                    max = score;
                                    bestLabel = label;
                                    argmax = h;
                                }
                            }
                        }

                        if(argmax!=goldHead ||  bestLabel.equals(goldLabel))  {
                            Object[] predictedFeatures=FeatureExtractor.extractFeatures(sentence, argmax, ch, bestLabel, dimension);
                            Object[] goldFeatures=FeatureExtractor.extractFeatures(sentence, goldHead, ch, goldLabel, dimension);

                            for(int i=0;i<predictedFeatures.length;i++){
                                if(predictedFeatures[i] instanceof String){
                                    if(!predictedFeatures[i].equals(goldFeatures[i])){
                                        perceptron.updateWeight(i,(String)predictedFeatures[i],-1.0);
                                        perceptron.updateWeight(i,(String)goldFeatures[i],1.0);
                                    }
                                }
                                else{
                                    HashMap<String,Integer> prd= (HashMap<String,Integer>)predictedFeatures[i];
                                    HashMap<String,Integer> gold= (HashMap<String,Integer>)goldFeatures[i];

                                    for(String feat:prd.keySet()){
                                        perceptron.updateWeight(i,feat,-prd.get(feat));
                                    }
                                    for(String feat:gold.keySet()){
                                        perceptron.updateWeight(i,feat,gold.get(feat));
                                    }
                                }
                            }
                        }     else{
                            correct++;
                        }

                        perceptron.incrementIteration();
                    }
                }
            }
            System.out.println("");
            double accuracy=100.0*correct/numDep;
            System.out.println("accuracy : "+accuracy);

            System.out.print("\nsaving current model...");
            perceptron.saveModel(modelPath+"_"+iter);
            System.out.println("done!");

            System.out.print("loading current model...");
           AveragedPerceptron avgPerceptron=AveragedPerceptron.loadModel(modelPath+"_"+iter);
            System.out.println("done!");

            System.out.print("\nParsing dev file...");

            int labelCorrect=0;
            int unlabelCorrect=0;
            int allDeps=0;
            senCount=0;
            for (Sentence sentence : devSentences) {
                senCount++;
                if(senCount%100==0){
                    System.out.print(senCount+"...");
                }
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch)) {
                        allDeps++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);

                        int argmax = 0;
                        String bestLabel = "";
                        double max = Double.NEGATIVE_INFINITY;

                        for (String label : possibleLabels) {
                            for (int h = 0; h < sentence.length(); h++) {
                                double score = avgPerceptron.score(FeatureExtractor.extractFeatures(sentence, h, ch, label, dimension), true);
                                if (score > max) {
                                    max = score;
                                    bestLabel = label;
                                    argmax = h;
                                }
                            }
                        }

                        if(argmax==goldHead){
                            unlabelCorrect++;
                            if(bestLabel.equals(goldLabel))
                                labelCorrect++;
                        }
                    }
                }
            }
            System.out.println("");

            double labeledAccuracy=100.0*labelCorrect/ allDeps;
            double unlabeledAccuracy=100.0*unlabelCorrect/ allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));
        }
    }

}
