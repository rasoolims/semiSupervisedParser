package Decoder;

import Structures.Sentence;

import java.util.ArrayList;
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
     * Currently it is based on McDonald et al. 2005 (Spanning Tree Methods for ...)
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

        features[index++] = hp + "|" + cp;
        features[index++] = hp + "|" + cp + "|" + distance;
        features[index++] = hp + "|" + cp + "|" + direction;

        features[index++] = hw + "|" + cw;
        features[index++] = hw + "|" + cw + "|" + distance;
        features[index++] = hw + "|" + cw + "|" + direction;

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
        if (headIndex < sentence.length() - 1)
            hNextP = sentence.pos(headIndex + 1);
        String hPrevP = "";
        if (headIndex > 0)
            hPrevP = sentence.pos(headIndex - 1);
        String cNextP = "";
        if (childIndex < sentence.length() - 1)
            cNextP = sentence.pos(childIndex + 1);
        String cPrevP = "";
        if (childIndex > 0)
            cPrevP = sentence.pos(childIndex - 1);

        features[index++] = hp + "|" + hNextP + "|" + cPrevP + "|" + cp;
        features[index++] = hPrevP + "|" + hp + "|" + cPrevP + "|" + cp;
        features[index++] = hp + "|" + hNextP + "|" + cp + "|" + cNextP;
        features[index++] = hPrevP + "|" + hp + "|" + cp + "|" + cNextP;

        if (!label.equals("")) {
            /**
             * From Table 1(b) in the paper
             */
            try {
                features[index++] = hwp + "|" + cwp + "|" + label;
                features[index++] = hwp + "|" + cwp + "|" + distance + "|" + label;
                features[index++] = hwp + "|" + cwp + "|" + direction + "|" + label;

            } catch (Exception ex) {
                System.err.println(label);
            }
            features[index++] = hp + "|" + cwp + "|" + label;
            features[index++] = hp + "|" + cwp + "|" + distance + "|" + label;
            features[index++] = hp + "|" + cwp + "|" + direction + "|" + label;

            features[index++] = hw + "|" + cwp + "|" + label;
            features[index++] = hw + "|" + cwp + "|" + distance + "|" + label;
            features[index++] = hw + "|" + cwp + "|" + direction + "|" + label;

            features[index++] = hwp + "|" + cp + "|" + label;
            features[index++] = hwp + "|" + cp + "|" + distance + "|" + label;
            features[index++] = hwp + "|" + cp + "|" + direction + "|" + label;

            features[index++] = hwp + "|" + cw + "|" + label;
            features[index++] = hwp + "|" + cw + "|" + distance + "|" + label;
            features[index++] = hwp + "|" + cw + "|" + direction + "|" + label;

            /**
             * From Table 1(c) in the paper
             */
            HashMap<String, Integer> inBetweenFeatures2 = new HashMap<String, Integer>();
            for (int i = Math.min(headIndex, childIndex) + 1; i < Math.max(headIndex, childIndex); i++) {
                String bp = sentence.pos(i);
                String mixFeat = hp + "|" + bp + "|" + cp + "|" + label;
                if (!inBetweenFeatures2.containsKey(maxFeatureLength))
                    inBetweenFeatures2.put(mixFeat, 1);
                else
                    inBetweenFeatures2.put(mixFeat, inBetweenFeatures2.get(mixFeat) + 1);
            }
            features[index++] = inBetweenFeatures2;

            features[index++] = hp + "|" + hNextP + "|" + cPrevP + "|" + cp + "|" + label;
            features[index++] = hPrevP + "|" + hp + "|" + cPrevP + "|" + cp + "|" + label;
            features[index++] = hp + "|" + hNextP + "|" + cp + "|" + cNextP + "|" + label;
            features[index++] = hPrevP + "|" + hp + "|" + cp + "|" + cNextP + "|" + label;
        }

        return features;
    }

    public static ArrayList<String> extract2ndOrderFeatures(Sentence sentence, int headIndex, int childIndex, int secondChildIndex) {
        ArrayList<String> features = new ArrayList<String>();
        if(true)
            return features;

        String cw1 = "_";
        String cp1 = "_";
        if (childIndex > 0) {
            cw1 = sentence.word(childIndex);
            cp1 = sentence.pos(childIndex);
        }
        String cwp1 = cw1 + "|" + cp1;

        String cw2 = sentence.word(secondChildIndex);
        String cp2 = sentence.pos(secondChildIndex);
        String cwp2 = cw2 + "|" + cp2;

        String hw = sentence.word(headIndex);
        String hp = sentence.pos(headIndex);
        String hwp = hw + "|" + hp;
        String direction = "l";
        int distance = 0;
        if (childIndex > 0)
            Math.abs(secondChildIndex - childIndex);
        if (secondChildIndex > headIndex)
            direction = "r";


        String tempFeat = "";

        // second order features
        tempFeat = "p_ijk:" + hp + "|" + cp1 + "|" + cp2;
        features.add(tempFeat);
        tempFeat = "p_ijk_dr:" + hp + "|" + cp1 + "|" + cp2 + "|" + direction;
        features.add(tempFeat);
        tempFeat = "p_ijk_d:" + hp + "|" + cp1 + "|" + cp2 + "|" + distance;
        features.add(tempFeat);

        tempFeat = "pp_jk:" + cp1 + "|" + cp2;
        features.add(tempFeat);
        tempFeat = "pp_jk_d:" + cp1 + "|" + cp2 + "|" + distance;
        features.add(tempFeat);
        tempFeat = "pp_jk_dr:" + cp1 + "|" + cp2 + "|" + direction;
        features.add(tempFeat);

        tempFeat = "pw_jk:" + cp1 + "|" + cw2;
        features.add(tempFeat);
        tempFeat = "pw_jk_d:" + cp1 + "|" + cw2 + "|" + distance;
        features.add(tempFeat);
        tempFeat = "pw_jk_dr:" + cp1 + "|" + cw2 + "|" + direction;
        features.add(tempFeat);

        tempFeat = "wp_jk:" + cw1 + "|" + cp2;
        features.add(tempFeat);
        tempFeat = "wp_jk_d:" + cw1 + "|" + cp2 + "|" + distance;
        features.add(tempFeat);
        tempFeat = "wp_jk_dr:" + cw1 + "|" + cp2 + "|" + direction;
        features.add(tempFeat);

        tempFeat = "ww_jk:" + cw1 + "|" + cw2;
        features.add(tempFeat);
        tempFeat = "ww_jk_d:" + cw1 + "|" + cw2 + "|" + distance;
        features.add(tempFeat);
        tempFeat = "ww_jk_dr:" + cw1 + "|" + cw2 + "|" + direction;
        features.add(tempFeat);

        return features;
    }

    public static ArrayList<String> extract1stOrderFeatures(Sentence sentence, int headIndex, int childIndex) {
        ArrayList<String> features = new ArrayList<String>();

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
        String tempFeat = "";
        tempFeat = "hwp:" + hwp;
        features.add(tempFeat);

        tempFeat = "hwp_d:" + hwp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hwp_dr:" + hwp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hw:" + hw;
        features.add(tempFeat);

        tempFeat = "hw_d:" + hw + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hw_dr:" + hw + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hp:" + hp;
        features.add(tempFeat);

        tempFeat = "hp_d:" + hp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hp_dr:" + hp + "|" + direction;
        features.add(tempFeat);


        tempFeat = "cwp:" + cwp;
        features.add(tempFeat);

        tempFeat = "cwp_d:" + cwp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "cwp_dr:" + cwp + "|" + direction;
        features.add(tempFeat);


        tempFeat = "cw:" + cw;
        features.add(tempFeat);

        tempFeat = "cw_d:" + cw + "|" + distance;
        features.add(tempFeat);

        tempFeat = "cw_dr:" + cw + "|" + direction;
        features.add(tempFeat);

        tempFeat = "cp:" + cp;
        features.add(tempFeat);

        tempFeat = "cp_d:" + cp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "cp_dr:" + cp + "|" + direction;
        features.add(tempFeat);

        /**
         * From Table 1(b) in the paper
         */
        tempFeat = "hwp_cwp:" + hwp + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hwp_cwp+d:" + hwp + "|" + cwp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hwp_cwp+dr:" + hwp + "|" + cwp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hp_cwp:" + hp + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hp_cwp_d:" + hp + "|" + cwp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hp_cwp_dr:" + hp + "|" + cwp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hw_cwp:" + hw + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hw_cwp_d:" + hw + "|" + cwp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hw_cwp_dr:" + hw + "|" + cwp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hwp_cp:" + hwp + "|" + cp;
        features.add(tempFeat);

        tempFeat = "hwp_cp_d:" + hwp + "|" + cp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hwp_cp_dr:" + hwp + "|" + cp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hwp_cw:" + hwp + "|" + cw;
        features.add(tempFeat);

        tempFeat = "hwp_cw_d:" + hwp + "|" + cw + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hwp_cw_dr:" + hwp + "|" + cw + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hp_cp:" + hp + "|" + cp;
        features.add(tempFeat);

        tempFeat = "hp_cp_d:" + hp + "|" + cp + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hp_cp_dr:" + hp + "|" + cp + "|" + direction;
        features.add(tempFeat);

        tempFeat = "hw_cw:" + hw + "|" + cw;
        features.add(tempFeat);

        tempFeat = "hw_cw_d:" + hw + "|" + cw + "|" + distance;
        features.add(tempFeat);

        tempFeat = "hw_cw_dr:" + hw + "|" + cw + "|" + direction;
        features.add(tempFeat);

        /**
         * From Table 1(c) in the paper
         */
        for (int i = Math.min(headIndex, childIndex) + 1; i < Math.max(headIndex, childIndex); i++) {
            String bp = sentence.pos(i);
            String mixFeat = "ib:" + hp + "|" + bp + "|" + cp;
            features.add(mixFeat);
        }

        String hNextP = "";
        if (headIndex < sentence.length() - 1)
            hNextP = sentence.pos(headIndex + 1);
        String hPrevP = "";
        if (headIndex > 0)
            hPrevP = sentence.pos(headIndex - 1);
        String cNextP = "";
        if (childIndex < sentence.length() - 1)
            cNextP = sentence.pos(childIndex + 1);
        String cPrevP = "";
        if (childIndex > 1)
            cPrevP = sentence.pos(childIndex - 1);

        tempFeat ="hncp:"+ hp + "|" + hNextP + "|" + cPrevP + "|" + cp;
        features.add(tempFeat);
        tempFeat ="hpcp:"+ hPrevP + "|" + hp + "|" + cPrevP + "|" + cp;
        features.add(tempFeat);
        tempFeat = "hncn:"+hp + "|" + hNextP + "|" + cp + "|" + cNextP;
        features.add(tempFeat);
        tempFeat ="hpcn:"+ hPrevP + "|" + hp + "|" + cp + "|" + cNextP;
        features.add(tempFeat);

        return features;
    }
}
