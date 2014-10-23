package Classifier;

import Decoder.GraphBasedParser;
import Structures.Sentence;
import com.sun.tools.javac.util.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 10/21/14
 * Time: 12:08 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class GenerativeModel implements Serializable {

    static HashSet<String> punctuations = new HashSet<String>();
    Pattern numPat ;

    private static void initializePuncs(){
        punctuations = new HashSet<String>();
        punctuations.add("#");
        punctuations.add("$");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add("[");
        punctuations.add("]");
        punctuations.add("{");
        punctuations.add("}");
        punctuations.add("\"");
        punctuations.add(",");
        punctuations.add(".");
        punctuations.add(":");
        punctuations.add("``");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add("-LSB-");
        punctuations.add("-RSB-");
        punctuations.add("-LCB-");
        punctuations.add("-RCB-");
    }

    int min=6;
    HashSet<String> wordCount;
    HashSet<String> posList;

    double wordSmoothing;

    double posSmoothing;

    /**
     * The key is the tuple <pos+word+dir+val>
     * The second key is the pos
     */
    HashMap<String,HashMap<String,Double>> fineGrainedPosCounts;

    HashMap<String,Integer> wordPosDirValCount;

    /**
     * The key is the tuple <pos+dir+val>
     * The second key is the pos
     */
    HashMap<String,HashMap<String,Double>> coarseGrainedPosCounts;

    HashMap<String,Integer> posDirValCount;


    /**
     * The key is the tuple <pos+word+dir+val>
     * The second key is the word
     */
    HashMap<String,HashMap<String,Double>> fineGrainedWordCounts;


    /**
     * The key is the tuple <pos+dir+val>
     * The second key is the word
     */
    HashMap<String,HashMap<String,Double>> coarseGrainedWordCounts;

    HashMap<String,HashMap<String,Double>> wordPosCount;
    HashMap<String,Integer> posCount;
    HashMap<String,HashMap<String,Double>> posPosCount;


    public GenerativeModel(double wordSmoothing,double posSmoothing){
        wordCount=new HashSet<String>();
        posList =new HashSet<String>();
        fineGrainedPosCounts =new HashMap<String, HashMap<String, Double>>();
        coarseGrainedPosCounts =new HashMap<String, HashMap<String, Double>>();
        fineGrainedWordCounts =new HashMap<String, HashMap<String, Double>>();
        coarseGrainedWordCounts =new HashMap<String, HashMap<String, Double>>();
        wordPosDirValCount=new HashMap<String, Integer>();
        posDirValCount=new HashMap<String, Integer>();
        this.wordSmoothing=wordSmoothing;
        this.posSmoothing=posSmoothing;
        wordPosCount=new HashMap<String, HashMap<String, Double>>();
        posCount=new HashMap<String, Integer>();
        posPosCount=new HashMap<String, HashMap<String, Double>>();
        numPat = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");
        initializePuncs();
    }

    public void createCounts(ArrayList<Sentence> trainSentences) {
        int tNum = 0;
        System.err.print("Creating counts...");
        HashMap<String, Integer> wc = new HashMap<String, Integer>();

        posList = new HashSet<String>();
        wordCount = new HashSet<String>();

        for (Sentence sentence : trainSentences) {
            for (int i = 0; i < sentence.length(); i++) {
                posList.add(sentence.pos(i));
                String word = sentence.word(i).toLowerCase();
                Matcher matcher=numPat.matcher(word);
                if(matcher.matches())
                    word="<num>";

                if (!wc.containsKey(word))
                    wc.put(word, 1);
                else
                    wc.put(word, 1 + wc.get(word));
            }
        }

        for (String w : wc.keySet()) {
            int count = wc.get(w);
            if (count >= min)
                wordCount.add(w);
        }

        for (Sentence sentence : trainSentences) {
            HashMap<Integer, Pair<TreeSet<Integer>, TreeSet<Integer>>> revDepDic = new HashMap<Integer, Pair<TreeSet<Integer>, TreeSet<Integer>>>();

            for (int i = 1; i < sentence.length(); i++) {
                int head = sentence.head(i);
                if (!revDepDic.containsKey(head)) {
                    revDepDic.put(head, new Pair<TreeSet<Integer>, TreeSet<Integer>>(new TreeSet<Integer>(), new TreeSet<Integer>()));
                }
                if (i < head)
                    (revDepDic.get(head)).fst.add(i);
                else
                    (revDepDic.get(head)).snd.add(i);
            }

            for (int i = 1; i < sentence.length(); i++) {
                if (!revDepDic.containsKey(i)) {
                    revDepDic.put(i, new Pair<TreeSet<Integer>, TreeSet<Integer>>(new TreeSet<Integer>(), new TreeSet<Integer>()));
                }
            }
            traverseTree(0, sentence, revDepDic);


            tNum++;
            if (tNum % 1000 == 0)
                System.err.print(tNum + "...");
        }
        System.err.print(tNum + "\n");

    }

    public double logProbability(Sentence sentence, int head, int mod, boolean val) {
       if(mod==0)
           return Double.NEGATIVE_INFINITY;
       if(Math.abs(head-mod)==1 && val)
           return      Double.NEGATIVE_INFINITY;

        String direction = (head > mod) ? "l" : "r";
        String valency = val ? "v" : "!v";

        String hw = sentence.word(head).toLowerCase();
        Matcher matcher = numPat.matcher(hw);
        if (matcher.matches())
            hw = "<num>";

        String hp = sentence.pos(head);
        if (!wordCount.contains(hw))
            hw = "UNKNOWN";

        String mp = "STOP";
        String mw = "STOP";
        if (mod >= 0 && mod < sentence.length()) {
            mw = sentence.word(mod).toLowerCase();
            matcher = numPat.matcher(mw);
            if (matcher.matches())
                mw = "<num>";
            mp = sentence.pos(mod);
            if (!wordCount.contains(mw))
                mw = "UNKNOWN";
        }

        if(mw.equals("STOP") && hp.equals("ROOT") && (direction.equals("l") || val))
            return 0;


        String wordPosDirVal = hw + "|" + hp + "|" + direction + "|" + valency;
        String posDirVal = hp + "|" + direction + "|" + valency;

        int f1 = 0;
        double n1 = 0;
        double n2 = 0;
        int f2 = 0;
        double n3 = 0;
        double n4 = 0;

        double n5 = 0;
        double n6 = 0;
        int f3 = 0;

        if (posCount.containsKey(hp)) {
            f3 = posCount.get(hp);

            if (wordPosCount.get(hp).containsKey(mw)) {
                n5 = wordPosCount.get(hp).get(mw);
            }

            if (posPosCount.get(hp).containsKey(mp)) {
                n6 = posPosCount.get(hp).get(mp);
            }
        }

        if (wordPosDirValCount.containsKey(wordPosDirVal)) {
            f1 = wordPosDirValCount.get(wordPosDirVal);
            n1 = wordSmoothing;
            n3 = posSmoothing;

            if (fineGrainedPosCounts.get(wordPosDirVal).containsKey(mp))
                n3 += fineGrainedPosCounts.get(wordPosDirVal).get(mp);

            if (fineGrainedWordCounts.get(wordPosDirVal).containsKey(mw))
                n1 += fineGrainedWordCounts.get(wordPosDirVal).get(mw);
        }

        int u1 = wordPosDirValCount.size();
        double l1 = (double) f1 / (f1 + 1 * u1);
        if (posDirValCount.containsKey(posDirVal)) {
            f2 = posDirValCount.get(posDirVal);
            n2 = wordSmoothing;
            n4 = posSmoothing;

            if (coarseGrainedWordCounts.get(posDirVal).containsKey(mw))
                n2 += coarseGrainedWordCounts.get(posDirVal).get(mw);
            if (coarseGrainedPosCounts.get(posDirVal).containsKey(mp))
                n4 += coarseGrainedPosCounts.get(posDirVal).get(mp);
        }
        int u2 = posDirValCount.size();
        double l2 = (double) f2 / (f2 + 5 * u2);

        double p = Math.log(l1 * n1 / (f1 + wordCount.size() * wordSmoothing) + (1.0 - l1) * (l2 * n2 / (f2 + posList.size() * posSmoothing) + (1.0 - l2) * n5 / f3)) + Math.log(l1 * n3 / (f1 + wordCount.size() * wordSmoothing) + (1 - l1) * (l2 * n4 / (f2 + posList.size() * posSmoothing) + (1. - l2) * n6 / f3));

        if(mw.equals("STOP"))
            p= Math.log(l1 * n3 / (f1 + wordCount.size() * wordSmoothing) + (1 - l1) * (l2 * n4 / (f2 + posList.size() * posSmoothing) + (1. - l2) * n6 / f3));
        return p;
    }

    public void parse(ArrayList<Sentence> sentences){
        System.err.print("parsing the sentences...");
        GraphBasedParser parser=new GraphBasedParser(this);
        int allDeps=0;
        double correctDeps=0.0;

        for(int i=0;i<sentences.size();i++){
            Sentence gSentence=sentences.get(i);
            Sentence pSentence= parser.eisner2ndOrder(gSentence);

            for(int j=1;j<pSentence.length();j++){
                if(!punctuations.contains(pSentence.pos(j))){
                    allDeps++;
                    if(pSentence.head(j)==gSentence.head(j))
                        correctDeps++;
                }
            }

            if((i+1)%100==0)
                System.err.print((i+1)+"...");
        }
        System.err.print("\n");

        double accuracy=100.0*correctDeps/ allDeps;
        System.err.println("accuracy:\t"+accuracy);

    }

    public static void saveModel(GenerativeModel generativeModel, String modelPath) throws Exception{
        ObjectOutput writer = new ObjectOutputStream(new FileOutputStream(modelPath));
        writer.writeObject(generativeModel);
        writer.flush();
        writer.close();
    }

    public static GenerativeModel loadModel(String modelPath) throws Exception{
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(modelPath));
        GenerativeModel gm= (GenerativeModel)reader.readObject();
        return gm;
    }

    private void traverseTree(int m, Sentence sentence,  HashMap<Integer,Pair<TreeSet<Integer>,TreeSet<Integer>>> revDepDic){
        String word=sentence.word(m).toLowerCase();
        Matcher matcher=numPat.matcher(word);
        if(matcher.matches())
            word="<num>";
        String pos=sentence.pos(m);

        if(!wordCount.contains(word))
            word="UNKNOWN";

        String stop="STOP";

        String leftWordPosDirNoVal=word+"|"+pos+"|l|!v";
        String rightWordPosDirNoVal=word+"|"+pos+"|r|!v";
        String leftPosDirNoVal=pos+"|l|!v";
        String rightPosDirNoVal=pos+"|r|!v";

        String leftWordPosDirVal=word+"|"+pos+"|l|v";
        String rightWordPosDirVal=word+"|"+pos+"|r|v";
        String leftPosDirVal=pos+"|l|v";
        String rightPosDirVal=pos+"|r|v";

        if(!wordPosDirValCount.containsKey(leftWordPosDirNoVal)) {
            wordPosDirValCount.put(leftWordPosDirNoVal, 0);
            fineGrainedPosCounts.put(leftWordPosDirNoVal, new HashMap<String, Double>());
            fineGrainedWordCounts.put(leftWordPosDirNoVal, new HashMap<String, Double>());
            fineGrainedPosCounts.get(leftWordPosDirNoVal).put(stop,0.0);
            fineGrainedWordCounts.get(leftWordPosDirNoVal).put(stop,0.0);
        }
        if(!wordPosDirValCount.containsKey(rightWordPosDirNoVal)) {
            wordPosDirValCount.put(rightWordPosDirNoVal, 0);
            fineGrainedPosCounts.put(rightWordPosDirNoVal, new HashMap<String, Double>());
            fineGrainedWordCounts.put(rightWordPosDirNoVal, new HashMap<String, Double>());
            fineGrainedPosCounts.get(rightWordPosDirNoVal).put(stop,0.0);
            fineGrainedWordCounts.get(rightWordPosDirNoVal).put(stop,0.0);
        }

        if(!posDirValCount.containsKey(leftPosDirNoVal)) {
            posDirValCount.put(leftPosDirNoVal, 0);
            coarseGrainedPosCounts.put(leftPosDirNoVal,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(leftPosDirNoVal,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(leftPosDirNoVal).put(stop,0.0);
            coarseGrainedWordCounts.get(leftPosDirNoVal).put(stop,0.0);
        }
        if(!posDirValCount.containsKey(rightPosDirNoVal)) {
            posDirValCount.put(rightPosDirNoVal, 0);
            coarseGrainedPosCounts.put(rightPosDirNoVal,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(rightPosDirNoVal,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(rightPosDirNoVal).put(stop,0.0);
            coarseGrainedWordCounts.get(rightPosDirNoVal).put(stop,0.0);
        }

        if(!wordPosDirValCount.containsKey(leftWordPosDirVal)) {
            wordPosDirValCount.put(leftWordPosDirVal, 0);
            fineGrainedPosCounts.put(leftWordPosDirVal, new HashMap<String, Double>());
            fineGrainedWordCounts.put(leftWordPosDirVal, new HashMap<String, Double>());
            fineGrainedPosCounts.get(leftWordPosDirVal).put(stop,0.0);
            fineGrainedWordCounts.get(leftWordPosDirVal).put(stop,0.0);
        }
        if(!wordPosDirValCount.containsKey(rightWordPosDirVal)) {
            wordPosDirValCount.put(rightWordPosDirVal, 0);
            fineGrainedPosCounts.put(rightWordPosDirVal, new HashMap<String, Double>());
            fineGrainedWordCounts.put(rightWordPosDirVal, new HashMap<String, Double>());
            fineGrainedPosCounts.get(rightWordPosDirVal).put(stop,0.0);
            fineGrainedWordCounts.get(rightWordPosDirVal).put(stop,0.0);
        }

        if(!posDirValCount.containsKey(leftPosDirVal)) {
            posDirValCount.put(leftPosDirVal, 0);
            coarseGrainedPosCounts.put(leftPosDirVal,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(leftPosDirVal,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(leftPosDirVal).put(stop,0.0);
            coarseGrainedWordCounts.get(leftPosDirVal).put(stop,0.0);
        }
        if(!posDirValCount.containsKey(rightPosDirVal)) {
            posDirValCount.put(rightPosDirVal, 0);
            coarseGrainedPosCounts.put(rightPosDirVal,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(rightPosDirVal,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(rightPosDirVal).put(stop,0.0);
            coarseGrainedWordCounts.get(rightPosDirVal).put(stop,0.0);
        }

        if(!posCount.containsKey(pos)){
            posCount.put(pos,0);
            posPosCount.put(pos,new HashMap<String, Double>());
            posPosCount.get(pos).put(stop,0.);
            wordPosCount.put(pos,new HashMap<String, Double>()) ;
        }

        posPosCount.get(pos).put(stop, posPosCount.get(pos).get(stop) + 2);
        posCount.put(pos, posCount.get(pos) + 2);

        //todo
        // traversing left children
        if(revDepDic.get(m).fst.size()==0){
            if(m!=0) {
                // stop  without getting any dependents on the left
                fineGrainedPosCounts.get(leftWordPosDirNoVal).put(stop, fineGrainedPosCounts.get(leftWordPosDirNoVal).get(stop) + 1);
                fineGrainedWordCounts.get(leftWordPosDirNoVal).put(stop, fineGrainedWordCounts.get(leftWordPosDirNoVal).get(stop) + 1);
                wordPosDirValCount.put(leftWordPosDirNoVal, wordPosDirValCount.get(leftWordPosDirNoVal) + 1);

                coarseGrainedPosCounts.get(leftPosDirNoVal).put(stop, coarseGrainedPosCounts.get(leftPosDirNoVal).get(stop) + 1);
                coarseGrainedWordCounts.get(leftPosDirNoVal).put(stop, coarseGrainedWordCounts.get(leftPosDirNoVal).get(stop) + 1);
                posDirValCount.put(leftPosDirNoVal, posDirValCount.get(leftPosDirNoVal) + 1);
            }
        }  else{
            boolean first=true;
            for(int mod:revDepDic.get(m).fst.descendingSet()){
                traverseTree(mod,sentence,revDepDic);
                String modWord=sentence.word(mod).toLowerCase();
                 matcher=numPat.matcher(modWord);
                if(matcher.matches())
                    modWord="<num>";
                if(!wordCount.contains(modWord))
                    modWord="UNKNOWN";
                String modPos=sentence.pos(mod);

                posCount.put(pos, posCount.get(pos) + 1);
                if(posPosCount.get(pos).containsKey(modPos))
                    posPosCount.get(pos).put(modPos,posPosCount.get(pos).get(modPos)+1);
                else
                    posPosCount.get(pos).put(modPos,1.);
                if(wordPosCount.get(pos).containsKey(modWord))
                    wordPosCount.get(pos).put(modWord,wordPosCount.get(pos).get(modWord)+1);
                else
                    wordPosCount.get(pos).put(modWord,1.);

                if(first){
                    if(!fineGrainedPosCounts.get(leftWordPosDirNoVal).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirNoVal).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirNoVal).put(modPos, fineGrainedPosCounts.get(leftWordPosDirNoVal).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(leftWordPosDirNoVal).containsKey(modWord))
                        fineGrainedWordCounts.get(leftWordPosDirNoVal).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(leftWordPosDirNoVal).put(modWord, fineGrainedWordCounts.get(leftWordPosDirNoVal).get(modWord)+1);

                    if(!coarseGrainedPosCounts.get(leftPosDirNoVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirNoVal).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirNoVal).put(modPos, coarseGrainedPosCounts.get(leftPosDirNoVal).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(leftPosDirNoVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(leftPosDirNoVal).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(leftPosDirNoVal).put(modWord, coarseGrainedWordCounts.get(leftPosDirNoVal).get(modWord)+1);

                    wordPosDirValCount.put(leftWordPosDirNoVal,wordPosDirValCount.get(leftWordPosDirNoVal)+1);
                    posDirValCount.put(leftPosDirNoVal,posDirValCount.get(leftPosDirNoVal)+1);
                }   else{
                    if(!fineGrainedPosCounts.get(leftWordPosDirVal).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirVal).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirVal).put(modPos, fineGrainedPosCounts.get(leftWordPosDirVal).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(leftWordPosDirVal).containsKey(modWord))
                        fineGrainedWordCounts.get(leftWordPosDirVal).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(leftWordPosDirVal).put(modWord, fineGrainedWordCounts.get(leftWordPosDirVal).get(modWord)+1);

                    if(!coarseGrainedPosCounts.get(leftPosDirVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirVal).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirVal).put(modPos, coarseGrainedPosCounts.get(leftPosDirVal).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(leftPosDirVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(leftPosDirVal).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(leftPosDirVal).put(modWord, coarseGrainedWordCounts.get(leftPosDirVal).get(modWord)+1);

                    wordPosDirValCount.put(leftWordPosDirVal,wordPosDirValCount.get(leftWordPosDirVal)+1);
                    posDirValCount.put(leftPosDirVal,posDirValCount.get(leftPosDirVal)+1);
                }
                first=false;
            }
            if(m!=0) {
                // stop  after getting dependents on the left
                fineGrainedPosCounts.get(leftWordPosDirVal).put(stop, fineGrainedPosCounts.get(leftWordPosDirVal).get(stop) + 1);
                fineGrainedWordCounts.get(leftWordPosDirVal).put(stop, fineGrainedWordCounts.get(leftWordPosDirVal).get(stop) + 1);
                wordPosDirValCount.put(leftWordPosDirVal, wordPosDirValCount.get(leftWordPosDirVal) + 1);

                coarseGrainedPosCounts.get(leftPosDirVal).put(stop, coarseGrainedPosCounts.get(leftPosDirVal).get(stop) + 1);
                coarseGrainedWordCounts.get(leftPosDirVal).put(stop, coarseGrainedWordCounts.get(leftPosDirVal).get(stop) + 1);
                posDirValCount.put(leftPosDirVal, posDirValCount.get(leftPosDirVal) + 1);
            }
        }


        // traversing right children
        if(revDepDic.get(m).snd.size()==0){
            if(m!=0) {
                // stop  without getting any dependents on the right
                fineGrainedPosCounts.get(rightWordPosDirNoVal).put(stop, fineGrainedPosCounts.get(rightWordPosDirNoVal).get(stop) + 1);
                fineGrainedWordCounts.get(rightWordPosDirNoVal).put(stop, fineGrainedWordCounts.get(rightWordPosDirNoVal).get(stop) + 1);
                wordPosDirValCount.put(rightWordPosDirNoVal, wordPosDirValCount.get(rightWordPosDirNoVal) + 1);

                coarseGrainedPosCounts.get(rightPosDirNoVal).put(stop, coarseGrainedPosCounts.get(rightPosDirNoVal).get(stop) + 1);
                coarseGrainedWordCounts.get(rightPosDirNoVal).put(stop, coarseGrainedWordCounts.get(rightPosDirNoVal).get(stop) + 1);
                posDirValCount.put(rightPosDirNoVal, posDirValCount.get(rightPosDirNoVal) + 1);
            }
        }  else{
            boolean first=true;
            for(int mod:revDepDic.get(m).snd){
                traverseTree(mod,sentence,revDepDic);
                String modWord=sentence.word(mod).toLowerCase();
                 matcher=numPat.matcher(modWord);
                if(matcher.matches())
                    modWord="<num>";
                if(!wordCount.contains(modWord))
                    modWord="UNKNOWN";
                String modPos=sentence.pos(mod);

                posCount.put(pos,posCount.get(pos)+1);
                if(posPosCount.get(pos).containsKey(modPos))
                    posPosCount.get(pos).put(modPos,posPosCount.get(pos).get(modPos)+1);
                else
                    posPosCount.get(pos).put(modPos,1.);
                if(wordPosCount.get(pos).containsKey(modWord))
                    wordPosCount.get(pos).put(modWord,wordPosCount.get(pos).get(modWord)+1);
                else
                    wordPosCount.get(pos).put(modWord,1.);

                if(first){
                    if(!fineGrainedPosCounts.get(rightWordPosDirNoVal).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirNoVal).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirNoVal).put(modPos, fineGrainedPosCounts.get(rightWordPosDirNoVal).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(rightWordPosDirNoVal).containsKey(modWord))
                        fineGrainedWordCounts.get(rightWordPosDirNoVal).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(rightWordPosDirNoVal).put(modWord, fineGrainedWordCounts.get(rightWordPosDirNoVal).get(modWord)+1);

                    if(!coarseGrainedPosCounts.get(rightPosDirNoVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirNoVal).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirNoVal).put(modPos, coarseGrainedPosCounts.get(rightPosDirNoVal).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(rightPosDirNoVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(rightPosDirNoVal).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(rightPosDirNoVal).put(modWord, coarseGrainedWordCounts.get(rightPosDirNoVal).get(modWord)+1);

                    wordPosDirValCount.put(rightWordPosDirNoVal,wordPosDirValCount.get(rightWordPosDirNoVal)+1);
                    posDirValCount.put(rightPosDirNoVal,posDirValCount.get(rightPosDirNoVal)+1);
                }   else{
                    if(!fineGrainedPosCounts.get(rightWordPosDirVal).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirVal).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirVal).put(modPos, fineGrainedPosCounts.get(rightWordPosDirVal).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(rightWordPosDirVal).containsKey(modWord))
                        fineGrainedWordCounts.get(rightWordPosDirVal).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(rightWordPosDirVal).put(modWord, fineGrainedWordCounts.get(rightWordPosDirVal).get(modWord)+1);


                    if(!coarseGrainedPosCounts.get(rightPosDirVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirVal).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirVal).put(modPos, coarseGrainedPosCounts.get(rightPosDirVal).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(rightPosDirVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(rightPosDirVal).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(rightPosDirVal).put(modWord, coarseGrainedWordCounts.get(rightPosDirVal).get(modWord)+1);

                    wordPosDirValCount.put(rightWordPosDirVal,wordPosDirValCount.get(rightWordPosDirVal)+1);
                    posDirValCount.put(rightPosDirVal,posDirValCount.get(rightPosDirVal)+1);
                }
                first=false;
            }

            if(m!=0) {
                // stop  after getting dependents on the right
                fineGrainedPosCounts.get(rightWordPosDirVal).put(stop, fineGrainedPosCounts.get(rightWordPosDirVal).get(stop) + 1);
                fineGrainedWordCounts.get(rightWordPosDirVal).put(stop, fineGrainedWordCounts.get(rightWordPosDirVal).get(stop) + 1);
                wordPosDirValCount.put(rightWordPosDirVal, wordPosDirValCount.get(rightWordPosDirVal) + 1);

                coarseGrainedPosCounts.get(rightPosDirVal).put(stop, coarseGrainedPosCounts.get(rightPosDirVal).get(stop) + 1);
                coarseGrainedWordCounts.get(rightPosDirVal).put(stop, coarseGrainedWordCounts.get(rightPosDirVal).get(stop) + 1);
                posDirValCount.put(rightPosDirVal, posDirValCount.get(rightPosDirVal) + 1);
            }
        }

    }

}
