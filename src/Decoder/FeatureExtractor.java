package Decoder;

import Structures.Sentence;

import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 4:34 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class FeatureExtractor {
    /**
     * This function returns the set of features
     * Currently it is based on McDonlad et al. 2005 (Spanning Tree Methods for ...)
     *
     * @param sentence
     * @param headIndex
     * @param childIndex
     * @param label
     * @param maxFeatureLength
     * @return
     */
    public static Object[] extractFeatures(Sentence sentence, int headIndex, int childIndex, String label, int maxFeatureLength) {
        Object[] features = new Object[maxFeatureLength];
        int index = 0;

        String cw = sentence.word(childIndex);
        String cp = sentence.pos(childIndex);
        String cwp = cw + "|" + cp;
        String hw = sentence.word(headIndex);
        String hp = sentence.pos(headIndex);
        String hwp = hw + "|" + hp;
        String direction = "l";
        int distance = Math.abs(headIndex - childIndex);
        if (childIndex > headIndex)
            direction = "r";

        /**
         * From Table 1(a) in the paper
         */
        features[index++] = hwp;
        features[index++] = hwp + "|" + distance;
        features[index++] = hwp + "|" + direction;

        features[index++] = hw;
        features[index++] = hw + "|" + distance;
        features[index++] = hw + "|" + direction;

        features[index++] = hp;
        features[index++] = hp + "|" + distance;
        features[index++] = hp + "|" + direction;

        features[index++] = cwp;
        features[index++] = cwp + "|" + distance;
        features[index++] = cwp + "|" + direction;

        features[index++] = cw;
        features[index++] = cw + "|" + distance;
        features[index++] = cw + "|" + direction;

        features[index++] = cp;
        features[index++] = cp + "|" + distance;
        features[index++] = cp + "|" + direction;

        /**
         * From Table 1(b) in the paper
         */
        features[index++] = hwp + "|" + cwp;
        features[index++] = hwp + "|" + cwp + "|" + distance;
        features[index++] = hwp + "|" + cwp + "|" + direction;

        features[index++] = hp + "|" + cwp;
        features[index++] = hp + "|" + cwp + "|" + distance;
        features[index++] = hp + "|" + cwp + "|" + direction;

        features[index++] = hw + "|" + cwp;
        features[index++] = hw + "|" + cwp + "|" + distance;
        features[index++] = hw + "|" + cwp + "|" + direction;

        features[index++] = hwp + "|" + cp;
        features[index++] = hwp + "|" + cp + "|" + distance;
        features[index++] = hwp + "|" + cp + "|" + direction;


        features[index++] = hwp + "|" + cw;
        features[index++] = hwp + "|" + cw + "|" + distance;
        features[index++] = hwp + "|" + cw + "|" + direction;

        features[index++] = hwp;
        features[index++] = hwp + "|" + distance;
        features[index++] = hwp + "|" + direction;

        features[index++] = cwp;
        features[index++] = cwp + "|" + distance;
        features[index++] = cwp + "|" + direction;

        /**
         * From Table 1(c) in the paper
         */
        HashMap<String, Integer> inBetweenFeatures = new HashMap<String, Integer>();
        for (int i = Math.min(headIndex, childIndex) + 1; i < Math.max(headIndex, childIndex); i++) {
            String bp = sentence.pos(i);
            String mixFeat = hp + "|" + bp + "|" + cp;
            if (!inBetweenFeatures.containsKey(maxFeatureLength))
                inBetweenFeatures.put(mixFeat, 1);
            else
                inBetweenFeatures.put(mixFeat, inBetweenFeatures.get(mixFeat) + 1);
        }
        features[index++] = inBetweenFeatures;

        String hNextP = "";
        if (headIndex < sentence.length()-1)
            hNextP = sentence.pos(headIndex + 1);
        String hPrevP = "";
        if (headIndex > 0)
            hPrevP = sentence.pos(headIndex - 1);
        String cNextP = "";
        if (childIndex < sentence.length()-1)
            cNextP = sentence.pos(childIndex + 1);
        String cPrevP = "";
        if (childIndex > 0)
            cPrevP = sentence.pos(childIndex - 1);

        features[index++]=hp+"|"+hNextP+"|"+cPrevP+"|"+cp;
        features[index++]=hPrevP+"|"+hp+"|"+cPrevP+"|"+cp;
        features[index++]=hp+"|"+hNextP+"|"+cp+"|"+cNextP;
        features[index++]=hPrevP+"|"+hp+"|"+cp+"|"+cNextP;


        return features;
    }
}
