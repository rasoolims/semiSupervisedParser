package Accessories;

import Structures.Sentence;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 3:27 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MSTReader {

    public static ArrayList<Sentence> readSentences(String path,boolean isRandom) throws Exception{
        BufferedReader reader=new BufferedReader(new FileReader(path));
        String line=null;

        Random random=new Random();
        ArrayList<Sentence> sentences=new ArrayList<Sentence>();

        while((line=reader.readLine())!=null){
            line=line.trim();
            if(line.length()<1)
                continue;

            String[] words=line.split("\t");
            line=reader.readLine();
            String[] posTags=line.split("\t");

            line=reader.readLine();
            String[] labels=line.split("\t");

            line=reader.readLine();
            String[] heads=line.split("\t");


            int length=words.length+1;
            String[] sWords=new String[length];
            String[] sTags=new String[length];
            int[] sHead=new int[length];
            String[] sLabels=new String[length];
            sWords[0]="ROOT";
            sTags[0]="ROOT";
            sHead[0]=-1;
            sLabels[0]="";

            for(int i=1;i<length;i++){
                 sWords[i]=words[i-1];
                sTags[i]=posTags[i-1];

                if(!isRandom || random.nextDouble()<0.2){
                    sHead[i]= Integer.parseInt(heads[i - 1]);
                   sLabels[i]=labels[i-1];
                }  else{
                    if(random.nextDouble()<1.0/8){
                        sHead[i]= random.nextInt(length);
                        sLabels[i]="";
                    }else{
                        sHead[i]= -1;
                        sLabels[i]="";
                    }
                }
            }

            Sentence sentence=new Sentence(sWords,sTags,sHead,sLabels);
            sentences.add(sentence);
        }
        return sentences;
    }

    public static ArrayList<Sentence> readRawSentences(String path) throws Exception{
        BufferedReader reader=new BufferedReader(new FileReader(path));
        String line=null;

        ArrayList<Sentence> sentences=new ArrayList<Sentence>();

        while((line=reader.readLine())!=null){
            line=line.trim();
            if(line.length()<1)
                continue;

            String[] words=line.split("\t");
            line=reader.readLine();
            String[] posTags=line.split("\t");

            int length=words.length+1;
            String[] sWords=new String[length];
            String[] sTags=new String[length];
            sWords[0]="ROOT";
            sTags[0]="ROOT";

            for(int i=1;i<length;i++){
                sWords[i]=words[i-1];
                sTags[i]=posTags[i-1];
            }

            Sentence sentence=new Sentence(sWords,sTags);
            sentences.add(sentence);
        }
        return sentences;
    }

}
