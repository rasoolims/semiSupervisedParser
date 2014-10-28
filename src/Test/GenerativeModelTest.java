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
     //    String trainPath= "/Users/msr/Desktop/train.auto.tag.mst";
     //   String devPath= "/Users/msr/Desktop/dev.auto.tag.mst";

    //     String trainPath= "/Users/msr/Desktop/full.mst";
      //    String devPath= "/Users/msr/Desktop/dev.auto.tag.mst";

       String trainPath= "/Users/msr/Desktop/train.p2m.mst";
      String devPath= "/Users/msr/Desktop/dev.p2m.mst";

        if(args.length>=2){
            trainPath=args[0];
            devPath=args[1];
        }

        ArrayList<Sentence> trainData = MSTReader.readSentences(trainPath, false);
        ArrayList<Sentence> devData = MSTReader.readSentences(devPath, false);
        GenerativeModel gm=new GenerativeModel(0,0);
        gm.createCounts(trainData);
    //    gm.parse(trainData);

        gm.parse(devData);
    }
}
