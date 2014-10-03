package Accessories;

import Structures.Sentence;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 3:27 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MSTReader {

    public static ArrayList<Sentence> readSentences(String path, boolean isWeighted) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line = null;

        int num_dep = 0;
        ArrayList<Sentence> sentences = new ArrayList<Sentence>();

        int sen_num = 0;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() < 1)
                continue;

            String[] words = line.split("\t");
            line = reader.readLine();
            String[] posTags = line.split("\t");

            line = reader.readLine();
            String[] labels = line.split("\t");

            line = reader.readLine();
            String[] heads = line.split("\t");


            int length = words.length + 1;
            String[] sWords = new String[length];
            double[] confidence = new double[length];
            String[] sTags = new String[length];
            int[] sHead = new int[length];
            String[] sLabels = new String[length];
            sWords[0] = "ROOT";
            sTags[0] = "ROOT";
            sHead[0] = -1;
            sLabels[0] = "";
            confidence[0]=0.0;

            for (int i = 1; i < length; i++) {
                sWords[i] = words[i - 1];
                if(sWords[i].equals("-LRB-"))
                    sWords[i]="(";
                if(sWords[i].equals("-RRB-"))
                    sWords[i]=")";
                sTags[i] = posTags[i - 1];

                confidence[i]=1.0;
                String[] spl=heads[i-1].split(":");
                int head=Integer.parseInt(spl[0]);
                if(spl.length>1 && isWeighted)
                    confidence[i]=Double.parseDouble(spl[1]);

                sHead[i] =head;
                sLabels[i] = labels[i - 1];
                if (sLabels[i].equals("_"))
                    sLabels[i] = "";
                if (head >= 0)
                    num_dep++;
            }

            Sentence sentence = new Sentence(sWords, sTags, sHead, sLabels,confidence);
            sentences.add(sentence);
            sen_num++;
            if(sen_num>3000)
                break;
            if (sen_num % 10000 == 0) {
                System.err.print(sen_num + "...");
            }
        }
        System.err.print("\nretrieved " + num_dep + " dependencies\n");

        return sentences;
    }

    public static ArrayList<Sentence> readRawSentences(String path) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line = null;

        ArrayList<Sentence> sentences = new ArrayList<Sentence>();

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() < 1)
                continue;

            String[] words = line.split("\t");
            line = reader.readLine();
            String[] posTags = line.split("\t");

            int length = words.length + 1;
            String[] sWords = new String[length];
            String[] sTags = new String[length];
            sWords[0] = "ROOT";
            sTags[0] = "ROOT";

            for (int i = 1; i < length; i++) {
                sWords[i] = words[i - 1];
                sTags[i] = posTags[i - 1];
            }

            Sentence sentence = new Sentence(sWords, sTags);
            sentences.add(sentence);
        }
        return sentences;
    }

}
