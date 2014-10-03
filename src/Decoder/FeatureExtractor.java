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

        String cw2 = sentence.word(secondChildIndex);
        String cp2 = sentence.pos(secondChildIndex);

        String hp = sentence.pos(headIndex);
        String direction = "l";
        int distance = 0;
        if (childIndex > 0)
            Math.abs(secondChildIndex - childIndex);
        if(distance>=10)
            distance=10;
        else if (distance>=5)
            distance=5;
        if (secondChildIndex > headIndex)
            direction = "r";


        String tempFeat = "";

        // second order features
        tempFeat = "2p_ijk:" + hp + "|" + cp1 + "|" + cp2;
        features.add(tempFeat);
     //   tempFeat = "p_ijk_dr:" + hp + "|" + cp1 + "|" + cp2 + "|" + direction;
     //   features.add(tempFeat);
        tempFeat = "2p_ijk_d:" + hp + "|" + cp1 + "|" + cp2 + "|" + distance+"|"+direction;
        features.add(tempFeat);

        tempFeat = "2pp_jk:" + cp1 + "|" + cp2;
        features.add(tempFeat);
        tempFeat = "2pp_jk_d:" + cp1 + "|" + cp2 + "|" + distance+"|"+direction;
        features.add(tempFeat);
      //  tempFeat = "pp_jk_dr:" + cp1 + "|" + cp2 + "|" + direction;
      //  features.add(tempFeat);

        tempFeat = "2pw_jk:" + cp1 + "|" + cw2;
        features.add(tempFeat);
        tempFeat = "2pw_jk_d:" + cp1 + "|" + cw2 + "|" + distance+"|"+direction;
        features.add(tempFeat);
       // tempFeat = "pw_jk_dr:" + cp1 + "|" + cw2 + "|" + direction;
      //  features.add(tempFeat);

        tempFeat = "2wp_jk:" + cw1 + "|" + cp2;
        features.add(tempFeat);
        tempFeat = "2wp_jk_d:" + cw1 + "|" + cp2 + "|" + distance+"|"+direction;
        features.add(tempFeat);
       // tempFeat = "wp_jk_dr:" + cw1 + "|" + cp2 + "|" + direction;
       // features.add(tempFeat);

        tempFeat = "2ww_jk:" + cw1 + "|" + cw2;
        features.add(tempFeat);
        tempFeat = "2ww_jk_d:" + cw1 + "|" + cw2 + "|" + distance+"|"+direction;
        features.add(tempFeat);
    //    tempFeat = "ww_jk_dr:" + cw1 + "|" + cw2 + "|" + direction;
     //   features.add(tempFeat);

        return features;
    }

    public static ArrayList<String> extract1stOrderFeatures(Sentence sentence, int headIndex, int childIndex) {
        ArrayList<String> features = new ArrayList<String>();

        String cw = sentence.word(childIndex);
        String cp = sentence.pos(childIndex);
        String cwp = cw + "|" + cp;
        String cPrefix =cw;
        String cpPrefix=cwp;
        if(cw.length()>5) {
            cPrefix = cw.substring(0, 5);
            cpPrefix=  cPrefix + "|" + cp;
        }

        String hw = sentence.word(headIndex);

        String hp = sentence.pos(headIndex);
        String hwp = hw + "|" + hp;
        String hPrefix =hw;
        String hpPrefix =hwp;
        if(hw.length()>5) {
            hPrefix = hw.substring(0, 5);
            hpPrefix=    hPrefix + "|" + hp;
        }

        int maxLen=Math.max(cw.length(),hw.length());

        //todo
        maxLen=0;
        //todo

        String direction = "l";
        int distance = Math.abs(headIndex - childIndex);
        if(distance>=10)
            distance=10;
        else if (distance>=5)
            distance=5;
        if (childIndex > headIndex)
            direction = "r";

        /**
         * From Table 1(a) in the paper
         */
        String tempFeat = "";
        tempFeat = "hwp:" + hwp;
        features.add(tempFeat);

        tempFeat = "hwp_d:" + hwp + "|" + distance +"|"+direction;
        features.add(tempFeat);

      //  tempFeat = "hwp_dr:" + hwp + "|" + direction;
     //   features.add(tempFeat);

        tempFeat = "hw:" + hw;
        features.add(tempFeat);

        tempFeat = "hw_d:" + hw + "|" + distance +"|"+direction;
        features.add(tempFeat);

    //    tempFeat = "hw_dr:" + hw + "|" + direction;
    //    features.add(tempFeat);

        tempFeat = "hp:" + hp;
        features.add(tempFeat);

        tempFeat = "hp_d:" + hp + "|" + distance +"|"+direction;
        features.add(tempFeat);

    //    tempFeat = "hp_dr:" + hp + "|" + direction;
     //   features.add(tempFeat);


        tempFeat = "cwp:" + cwp;
        features.add(tempFeat);

        tempFeat = "cwp_d:" + cwp + "|" + distance +"|"+direction;
        features.add(tempFeat);

    //    tempFeat = "cwp_dr:" + cwp + "|" + direction;
    //    features.add(tempFeat);


        tempFeat = "cw:" + cw;
        features.add(tempFeat);

        tempFeat = "cw_d:" + cw + "|" + distance +"|"+direction;
        features.add(tempFeat);

     //   tempFeat = "cw_dr:" + cw + "|" + direction;
     //   features.add(tempFeat);

        tempFeat = "cp:" + cp;
        features.add(tempFeat);

        tempFeat = "cp_d:" + cp + "|" + distance +"|"+direction;
        features.add(tempFeat);

     //   tempFeat = "cp_dr:" + cp + "|" + direction;
    //    features.add(tempFeat);

        if(maxLen>5){
            if(hw.length()>5) {
                tempFeat = "hwp:" + hpPrefix;
                features.add(tempFeat);

                tempFeat = "hwp_d:" + hpPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

               //   tempFeat = "hwp_dr:" + hpPrefix + "|" + direction;
              //    features.add(tempFeat);

                tempFeat = "hw:" + hPrefix;
                features.add(tempFeat);

                tempFeat = "hw_d:" + hPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

              //    tempFeat = "hw_dr:" + hPrefix + "|" + direction;
              //    features.add(tempFeat);
            }

            if(cw.length()>5) {
                tempFeat = "cwp:" + cpPrefix;
                features.add(tempFeat);

                tempFeat = "cwp_d:" + cpPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

             //     tempFeat = "cwp_dr:" + cpPrefix + "|" + direction;
             //     features.add(tempFeat);


                tempFeat = "cw:" + cPrefix;
                features.add(tempFeat);

                tempFeat = "cw_d:" + cPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

             //     tempFeat = "cw_dr:" + cPrefix + "|" + direction;
             //     features.add(tempFeat);
            }
        }

        /**
         * From Table 1(b) in the paper
         */
        tempFeat = "hwp_cwp:" + hwp + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hwp_cwp+d:" + hwp + "|" + cwp + "|" + distance +"|"+direction;
        features.add(tempFeat);

      //  tempFeat = "hwp_cwp+dr:" + hwp + "|" + cwp + "|" + direction;
     //   features.add(tempFeat);

        tempFeat = "hp_cwp:" + hp + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hp_cwp_d:" + hp + "|" + cwp + "|" + distance +"|"+direction;
        features.add(tempFeat);

    //    tempFeat = "hp_cwp_dr:" + hp + "|" + cwp + "|" + direction;
    //    features.add(tempFeat);

        tempFeat = "hw_cwp:" + hw + "|" + cwp;
        features.add(tempFeat);

        tempFeat = "hw_cwp_d:" + hw + "|" + cwp + "|" + distance +"|"+direction;
        features.add(tempFeat);

    //    tempFeat = "hw_cwp_dr:" + hw + "|" + cwp + "|" + direction;
     //   features.add(tempFeat);

        tempFeat = "hwp_cp:" + hwp + "|" + cp;
        features.add(tempFeat);

        tempFeat = "hwp_cp_d:" + hwp + "|" + cp + "|" + distance +"|"+direction;
        features.add(tempFeat);

   //     tempFeat = "hwp_cp_dr:" + hwp + "|" + cp + "|" + direction;
    //    features.add(tempFeat);

        tempFeat = "hwp_cw:" + hwp + "|" + cw;
        features.add(tempFeat);

        tempFeat = "hwp_cw_d:" + hwp + "|" + cw + "|" + distance +"|"+direction;
        features.add(tempFeat);

     //   tempFeat = "hwp_cw_dr:" + hwp + "|" + cw + "|" + direction;
    //    features.add(tempFeat);

        tempFeat = "hp_cp:" + hp + "|" + cp;
        features.add(tempFeat);

        tempFeat = "hp_cp_d:" + hp + "|" + cp + "|" + distance +"|"+direction;
        features.add(tempFeat);

     //   tempFeat = "hp_cp_dr:" + hp + "|" + cp + "|" + direction;
      //  features.add(tempFeat);

        tempFeat = "hw_cw:" + hw + "|" + cw;
        features.add(tempFeat);

        tempFeat = "hw_cw_d:" + hw + "|" + cw + "|" + distance +"|"+direction;
        features.add(tempFeat);

     //   tempFeat = "hw_cw_dr:" + hw + "|" + cw + "|" + direction;
    //    features.add(tempFeat);

        if(maxLen>5){
            if(hw.length()>5) {
                tempFeat = "hwp_cwp:" + hpPrefix + "|" + cpPrefix;
                features.add(tempFeat);

                tempFeat = "hwp_cwp+d:" + hpPrefix + "|" + cpPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

              //    tempFeat = "hwp_cwp+dr:" + hpPrefix + "|" + cpPrefix + "|" + direction;
              //    features.add(tempFeat);
            }

            if(cw.length()>5) {
                tempFeat = "hp_cwp:" + hp + "|" + cpPrefix;
                features.add(tempFeat);

                tempFeat = "hp_cwp_d:" + hp + "|" + cpPrefix + "|" + distance + "|" + direction;
                features.add(tempFeat);

              //    tempFeat = "hp_cwp_dr:" + hp + "|" + cpPrefix + "|" + direction;
              //    features.add(tempFeat);
            }

            tempFeat = "hw_cwp:" + hPrefix + "|" + cpPrefix;
            features.add(tempFeat);

            tempFeat = "hw_cwp_d:" + hPrefix + "|" + cpPrefix + "|" + distance +"|"+direction;
            features.add(tempFeat);

         //     tempFeat = "hw_cwp_dr:" + hPrefix + "|" + cpPrefix + "|" + direction;
         //     features.add(tempFeat);

            tempFeat = "hwp_cp:" + hpPrefix + "|" + cp;
            features.add(tempFeat);

            tempFeat = "hwp_cp_d:" + hpPrefix + "|" + cp + "|" + distance +"|"+direction;
            features.add(tempFeat);

        //      tempFeat = "hwp_cp_dr:" + hpPrefix + "|" + cp + "|" + direction;
        //      features.add(tempFeat);

            tempFeat = "hwp_cw:" + hpPrefix + "|" + cPrefix;
            features.add(tempFeat);

            tempFeat = "hwp_cw_d:" + hpPrefix + "|" + cPrefix + "|" + distance +"|"+direction;
            features.add(tempFeat);

        //       tempFeat = "hwp_cw_dr:" + hpPrefix + "|" + cPrefix + "|" + direction;
        //       features.add(tempFeat);

            tempFeat = "hw_cw:" + hPrefix + "|" + cPrefix;
            features.add(tempFeat);

            tempFeat = "hw_cw_d:" + hPrefix + "|" + cPrefix + "|" + distance +"|"+direction;
            features.add(tempFeat);

        //      tempFeat = "hw_cw_dr:" + hPrefix + "|" + cPrefix + "|" + direction;
        //      features.add(tempFeat);
        }


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
