package Classifier;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 10/10/14
 * Time: 3:32 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class AdaGrad extends OnlineClassifier{
    HashMap<String, Double> weights;
    double learningRate;
    double ridge;

    HashMap<String,Double> gDiag;

    public AdaGrad(){
        weights=new HashMap<String, Double>(1000000);
        gDiag=new HashMap<String, Double>(1000000);
        iteration=1;
    }

    public AdaGrad(double learningRate,double ridge){
        this();
        this.learningRate=learningRate;
        this.ridge=ridge;
    }

    @Override
    public void updateWeight(String feature, double change) {
       if(gDiag.containsKey(feature))
           gDiag.put(feature,gDiag.get(feature)+Math.pow(change,2));
       else
           gDiag.put(feature,Math.pow(change,2));

       double newValue=0;
       if(weights.containsKey(feature))
           newValue=weights.get(feature);

      newValue+=learningRate*(1.0/(ridge+Math.sqrt(gDiag.get(feature))))*change;

      weights.put(feature,newValue);
    }

    @Override
    public double score(ArrayList<String> features, boolean decode) {
        double value=0.0;
        for(String feat:features)
            if(weights.containsKey(feat))
                value+=weights.get(feat);
        return value;
    }

    @Override
    public void saveModel(String modelPath) throws Exception {
        ObjectOutput writer = new ObjectOutputStream(new FileOutputStream(modelPath));
        writer.writeObject(weights);
        writer.flush();
        writer.close();
    }

    @Override
    public int size() {
        return weights.size();
    }

    @Override
    public OnlineClassifier loadModel(String modelPath) throws Exception {
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(modelPath));
        HashMap<String, Double> weights = (HashMap<String, Double>) reader.readObject();

        AdaGrad adaGrad = new AdaGrad();
        adaGrad.setWeights(weights);

        return adaGrad;
    }

    public void setWeights(HashMap<String, Double> weights) {
        this.weights = weights;
    }
}
