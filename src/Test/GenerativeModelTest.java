package Test;

import Accessories.MSTReader;
import Classifier.GenerativeModel;
import Structures.Sentence;

import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 10/21/14
 * Time: 12:22 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class GenerativeModelTest {
    public static void main(String[] args) throws Exception {
        String trainPath= "/Users/msr/Desktop/full.train.mst";
        if(args.length>=1){
            trainPath=args[0];
        }
        ArrayList<Sentence> trainData = MSTReader.readSentences(trainPath, false);
        GenerativeModel gm=new GenerativeModel(.01,.001);
        gm.createCounts(trainData);
    }
}
