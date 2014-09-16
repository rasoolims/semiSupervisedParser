package Classifer;

import java.io.*;
import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 11:26 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class AveragedPerceptron implements Serializable{
    Object[] weights;
    Object[] avgWeights;
    int iteration;

    public AveragedPerceptron(int length){
        weights=new Object[length];
        avgWeights=new Object[length];

        for(int i=0;i<length;i++){
            weights[i]=new HashMap<String, Double>();
            avgWeights[i]=new HashMap<String, Double>();
        }
        iteration=1;
    }

    public void updateWeight(int slotNum, String feature, double change){
        HashMap<String, Double> map=(HashMap<String, Double>)weights[slotNum];
        HashMap<String, Double> avgMap=(HashMap<String, Double>)avgWeights[slotNum];

        if(!map.containsKey(feature)){
            map.put(feature,change);
        }  else{
            map.put(feature,map.get(feature)+change);
        }

        if(!avgMap.containsKey(feature)){
            avgMap.put(feature,iteration*change);
        }  else{
            avgMap.put(feature,avgMap.get(feature)+iteration*change);
        }
    }

    public void incrementIteration(){
        iteration++;
    }

    public void saveModel(String modelPath) throws  Exception{
        Object[] finalAverageWeight=new Object[weights.length];
        for(int i=0;i<weights.length;i++){
            HashMap<String, Double> map=(HashMap<String, Double>)weights[i];
            HashMap<String, Double> avgMap=(HashMap<String, Double>)avgWeights[i];
            finalAverageWeight[i]=new HashMap<String, Double>();

            for(String feat:map.keySet()){
                double newValue=  map.get(feat)-(avgMap.get(feat)/iteration);
                if(newValue!=0.0)
                ((HashMap<String, Double>)finalAverageWeight[i]).put(feat,newValue);
            }
        }

        ObjectOutput writer = new ObjectOutputStream(new FileOutputStream(modelPath));
        writer.writeObject(finalAverageWeight);
        writer.flush();
        writer.close();
    }

    public static AveragedPerceptron loadModel(String modelPath) throws  Exception {
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(modelPath));
        Object[] avgWeights= (Object[])reader.readObject();

        AveragedPerceptron averagedPerceptron=new AveragedPerceptron(avgWeights.length);
        averagedPerceptron.avgWeights=avgWeights;

        return averagedPerceptron;
    }

    public double weight(int slotNum,String feat, boolean decode){
        if(!decode){
            HashMap<String, Double> map=(HashMap<String, Double>)weights[slotNum];
            if(map.containsKey(feat))
                return map.get(feat);
            else
                return 0;
        }   else{
                HashMap<String, Double> map=(HashMap<String, Double>)avgWeights[slotNum];
                if(map.containsKey(feat))
                    return map.get(feat);
                else
                    return 0;
        }
    }

    public double score(Object[] features,boolean decode){
        double score=0.0;
        for(int i=0;i<features.length;i++){
            HashMap<String,Double> map;
            if(decode)
                map= (HashMap<String,Double>)avgWeights[i];
            else
                map= (HashMap<String,Double>)weights[i];
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

    public int dimension(){
        return avgWeights.length;
    }

    public int size(){
        int size=0;
        for(int i=0;i<avgWeights.length;i++)
            size+=((HashMap<String,Double>)avgWeights[i]).size();
        return size;
    }
}
