package Classifier;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 11:26 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class AveragedPerceptron implements Serializable{
    HashMap<String, Double> weights;
    HashMap<String, Double> avgWeights;
    int iteration;


    public AveragedPerceptron() {
        weights = new HashMap<String, Double>(1000000);
        avgWeights = new HashMap<String, Double>(1000000);
        iteration = 1;
    }

    public void updateWeight(String feature, double change){
        if(!weights.containsKey(feature)){
            weights.put(feature,change);
        }  else{
            weights.put(feature,weights.get(feature)+change);
        }

        if(!avgWeights.containsKey(feature)){
            avgWeights.put(feature,iteration*change);
        }  else{
            avgWeights.put(feature,avgWeights.get(feature)+iteration*change);
        }
    }

    public void incrementIteration(){
        iteration++;
    }

    public void saveModel(String modelPath) throws  Exception{
        HashMap<String, Double> finalAverageWeight=new  HashMap<String, Double>(avgWeights.size());

            for(String feat:weights.keySet()){
                double newValue=  weights.get(feat)-(avgWeights.get(feat)/iteration);
                if(newValue!=0.0)
                finalAverageWeight.put(feat,newValue);
            }
        ObjectOutput writer = new ObjectOutputStream(new FileOutputStream(modelPath));
        writer.writeObject(finalAverageWeight);
        writer.flush();
        writer.close();
    }

    public static AveragedPerceptron loadModel(String modelPath) throws  Exception {
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(modelPath));
        HashMap<String, Double> avgWeights= (HashMap<String, Double>)reader.readObject();

        AveragedPerceptron averagedPerceptron=new AveragedPerceptron();
        averagedPerceptron.avgWeights=avgWeights;

        return averagedPerceptron;
    }

    public double weight(int slotNum,String feat, boolean decode){
        if(!decode){
            if(weights.containsKey(feat))
                return weights.get(feat);
            else
                return 0;
        }   else{
                if(avgWeights.containsKey(feat))
                    return avgWeights.get(feat);
                else
                    return 0;
        }
    }

    public double score(Object[] features,boolean decode){
        double score=0.0;
        for(int i=0;i<features.length;i++){
            HashMap<String,Double> map;
            if(decode)
                map= avgWeights;
            else
                map= weights;
            if(features[i] instanceof String){
                if(map.containsKey(features[i]))
                     score+=map.get(features[i]);
            }
            else{
                // this is just for the in-between features
                HashMap<String,Integer> feats=(HashMap<String,Integer>)features[i];

                for(String feat:feats.keySet()){
                    if(map.containsKey(feat))
                    score+=feats.get(feat)*map.get(feat);
                }
            }
        }
        return score;
    }


    public double score(ArrayList<String> features,boolean decode){
        double score=0.0;
            HashMap<String,Double> map;
            if(decode)
                map= avgWeights;
            else
                map= weights;
        for(String feature:features){
            if(map.containsKey(feature))
                    score+=map.get(feature);
        }
        return score;
    }


    public int size(){
        return avgWeights.size();
    }
}
