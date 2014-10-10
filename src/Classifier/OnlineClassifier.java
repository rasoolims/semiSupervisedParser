package Classifier;

import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 10/10/14
 * Time: 12:05 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class OnlineClassifier {
    public abstract void updateWeight(String feature, double change);
    public abstract  double score(ArrayList<String> features,boolean decode);
    public abstract void saveModel(String modelPath)  throws  Exception;
    public  abstract void incrementIteration();
    public abstract int size();
    public abstract OnlineClassifier loadModel(String modelPath) throws  Exception;
}
