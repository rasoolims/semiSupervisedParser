package Classifier;

import Structures.Sentence;
import com.sun.tools.javac.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 10/21/14
 * Time: 12:08 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class GenerativeModel {
    HashSet<String> wordCount;
    HashSet<String> posCount;

    double wordSmoothing;

    double posSmoothing;

    /**
     * The key is the tuple <pos+word+dir+adj>
     * The second key is the pos
     */
    HashMap<String,HashMap<String,Double>> fineGrainedPosCounts;

    HashMap<String,Integer> wordPosDirAdjCount;

    /**
     * The key is the tuple <pos+dir+adj>
     * The second key is the pos
     */
    HashMap<String,HashMap<String,Double>> coarseGrainedPosCounts;

    HashMap<String,Integer> posDirAdjCount;


    /**
     * The key is the tuple <pos+word+dir+adj>
     * The second key is the word
     */
    HashMap<String,HashMap<String,Double>> fineGrainedWordCounts;


    /**
     * The key is the tuple <pos+dir+adj>
     * The second key is the word
     */
    HashMap<String,HashMap<String,Double>> coarseGrainedWordCounts;

    public GenerativeModel(double wordSmoothing,double posSmoothing){
        wordCount=new HashSet<String>();
        posCount=new HashSet<String>();
        fineGrainedPosCounts =new HashMap<String, HashMap<String, Double>>();
        coarseGrainedPosCounts =new HashMap<String, HashMap<String, Double>>();
        fineGrainedWordCounts =new HashMap<String, HashMap<String, Double>>();
        coarseGrainedWordCounts =new HashMap<String, HashMap<String, Double>>();
        wordPosDirAdjCount=new HashMap<String, Integer>();
        posDirAdjCount=new HashMap<String, Integer>();
        this.wordSmoothing=wordSmoothing;
        this.posSmoothing=posSmoothing;
    }

    public void createCounts(ArrayList<Sentence> trainSentences){
       int tNum=0;
        System.err.print("Creating counts...");
        HashMap<String,Integer> wc=new HashMap<String, Integer>();

        posCount=new HashSet<String>();
        wordCount=new HashSet<String>();

        for(Sentence sentence:trainSentences) {
            for (int i = 0; i < sentence.length(); i++) {
                posCount.add(sentence.pos(i));
                String word = sentence.word(i);
                if(!wc.containsKey(word))
                    wc.put(word,1);
                else
                    wc.put(word,1+wc.get(word));
            }
        }

        for(String w :wc.keySet()){
            int count=wc.get(w);
            if(count>=5)
                wordCount.add(w);
        }

      for(Sentence sentence:trainSentences) {
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
        System.err.print(tNum+"\n");

        //sanity check
        Sentence sentence=trainSentences.get(0);
        System.err.println(probability(sentence,0,2,false));
        System.err.println(probability(sentence,0,1,false));
        System.err.println(probability(sentence,2,5,true));
        System.err.println(probability(sentence,2,4,false));
        System.err.println(probability(sentence,4,2,false));
        System.err.println(probability(sentence,4,3,false));
    }

    public double probability(Sentence sentence,int head, int mod, boolean adj){
        String direction=(head>mod)?"l":"r";
        String adjacency=adj?"a":"!a";

        String hw=sentence.word(head);
        String hp=sentence.pos(head);
        if(!wordCount.contains(hw))
            hw="UNKNOWN";

        String mw=sentence.word(mod);
        String mp=sentence.pos(mod);
        if(!wordCount.contains(mw))
            mw="UNKNOWN";

        String wordPosDirAdj=hw+"|"+hp+"|"+direction+"|"+adjacency;
        String posDirAdj=hp+"|"+direction+"|"+adjacency;

        int f1=0;
        double n1=wordSmoothing;
        double n2=wordSmoothing;
        int f2=0;
        double n3=posSmoothing;
        double n4=posSmoothing;

        if(wordPosDirAdjCount.containsKey(wordPosDirAdj)) {
            f1 = wordPosDirAdjCount.get(wordPosDirAdj);
            if(fineGrainedPosCounts.get(wordPosDirAdj).containsKey(mp))
                n3+=fineGrainedPosCounts.get(wordPosDirAdj).get(mp);

            if(fineGrainedWordCounts.get(wordPosDirAdj).containsKey(mw))
                n1+=fineGrainedWordCounts.get(wordPosDirAdj).get(mw);

        }
        int u1=wordPosDirAdjCount.size();
        double l1=(double)f1/(f1+5*u1);


        if(posDirAdjCount.containsKey(posDirAdj)) {
            f2 = posDirAdjCount.get(posDirAdj);
            if(coarseGrainedWordCounts.get(posDirAdj).containsKey(mw))
                n2+=coarseGrainedWordCounts.get(posDirAdj).get(mw);
            if(coarseGrainedPosCounts.get(posDirAdj).containsKey(mp))
                n4+=coarseGrainedPosCounts.get(posDirAdj).get(mp);
        }
        int u2= posDirAdjCount.size();
        double l2=(double)f2/(f2+5*u2);

        double p=Math.log(l1*n1/(f1+wordCount.size()*wordSmoothing) + l2*n2/(f2+posCount.size()*posSmoothing))+Math.log(l1*n3/(f1+wordCount.size()*wordSmoothing) + l2*n4/(f2+posCount.size()*posSmoothing));

        return p;
    }

    private void traverseTree(int m, Sentence sentence,  HashMap<Integer,Pair<TreeSet<Integer>,TreeSet<Integer>>> revDepDic){
        String word=sentence.word(m);
        String pos=sentence.pos(m);

        if(!wordCount.contains(word))
            word="UNKNOWN";

        String stop="STOP";

        String leftWordPosDirNoAdj=word+"|"+pos+"|l|!a";
        String rightWordPosDirNoAdj=word+"|"+pos+"|r|!a";
        String leftPosDirNoAdj=pos+"|l|!a";
        String rightPosDirNoAdj=pos+"|r|!a";

        String leftWordPosDirAdj=word+"|"+pos+"|l|a";
        String rightWordPosDirAdj=word+"|"+pos+"|r|a";
        String leftPosDirAdj=pos+"|l|a";
        String rightPosDirAdj=pos+"|r|a";

        if(!wordPosDirAdjCount.containsKey(leftWordPosDirNoAdj)) {
            wordPosDirAdjCount.put(leftWordPosDirNoAdj, 0);
            fineGrainedPosCounts.put(leftWordPosDirNoAdj, new HashMap<String, Double>());
            fineGrainedWordCounts.put(leftWordPosDirNoAdj, new HashMap<String, Double>());
            fineGrainedPosCounts.get(leftWordPosDirNoAdj).put(stop,0.0);
        }
        if(!wordPosDirAdjCount.containsKey(rightWordPosDirNoAdj)) {
            wordPosDirAdjCount.put(rightWordPosDirNoAdj, 0);
            fineGrainedPosCounts.put(rightWordPosDirNoAdj, new HashMap<String, Double>());
            fineGrainedWordCounts.put(rightWordPosDirNoAdj, new HashMap<String, Double>());
            fineGrainedPosCounts.get(rightWordPosDirNoAdj).put(stop,0.0);
        }

        if(!posDirAdjCount.containsKey(leftPosDirNoAdj)) {
            posDirAdjCount.put(leftPosDirNoAdj, 0);
            coarseGrainedPosCounts.put(leftPosDirNoAdj,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(leftPosDirNoAdj,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(leftPosDirNoAdj).put(stop,0.0);
        }
        if(!posDirAdjCount.containsKey(rightPosDirNoAdj)) {
            posDirAdjCount.put(rightPosDirNoAdj, 0);
            coarseGrainedPosCounts.put(rightPosDirNoAdj,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(rightPosDirNoAdj,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(rightPosDirNoAdj).put(stop,0.0);
        }

        if(!wordPosDirAdjCount.containsKey(leftWordPosDirAdj)) {
            wordPosDirAdjCount.put(leftWordPosDirAdj, 0);
            fineGrainedPosCounts.put(leftWordPosDirAdj, new HashMap<String, Double>());
            fineGrainedWordCounts.put(leftWordPosDirAdj, new HashMap<String, Double>());
            fineGrainedPosCounts.get(leftWordPosDirAdj).put(stop,0.0);
        }
        if(!wordPosDirAdjCount.containsKey(rightWordPosDirAdj)) {
            wordPosDirAdjCount.put(rightWordPosDirAdj, 0);
            fineGrainedPosCounts.put(rightWordPosDirAdj, new HashMap<String, Double>());
            fineGrainedWordCounts.put(rightWordPosDirAdj, new HashMap<String, Double>());
            fineGrainedPosCounts.get(rightWordPosDirAdj).put(stop,0.0);
        }

        if(!posDirAdjCount.containsKey(leftPosDirAdj)) {
            posDirAdjCount.put(leftPosDirAdj, 0);
            coarseGrainedPosCounts.put(leftPosDirAdj,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(leftPosDirAdj,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(leftPosDirAdj).put(stop,0.0);
        }
        if(!posDirAdjCount.containsKey(rightPosDirAdj)) {
            posDirAdjCount.put(rightPosDirAdj, 0);
            coarseGrainedPosCounts.put(rightPosDirAdj,new  HashMap<String, Double>());
            coarseGrainedWordCounts.put(rightPosDirAdj,new  HashMap<String, Double>());
            coarseGrainedPosCounts.get(rightPosDirAdj).put(stop,0.0);
        }

        //todo
        // traversing left children
        if(revDepDic.get(m).fst.size()==0){
            // stop  without getting any dependents on the left
            fineGrainedPosCounts.get(leftWordPosDirNoAdj).put(stop, fineGrainedPosCounts.get(leftWordPosDirNoAdj).get(stop)+1);
            wordPosDirAdjCount.put(leftWordPosDirNoAdj,wordPosDirAdjCount.get(leftWordPosDirNoAdj)+1);

            coarseGrainedPosCounts.get(leftPosDirNoAdj).put(stop,coarseGrainedPosCounts.get(leftPosDirNoAdj).get(stop)+1);
            posDirAdjCount.put(leftPosDirNoAdj,posDirAdjCount.get(leftPosDirNoAdj)+1);
        }  else{
            boolean first=true;
            for(int mod:revDepDic.get(m).fst){
                traverseTree(mod,sentence,revDepDic);
                String modWord=sentence.word(mod);
                if(!wordCount.contains(modWord))
                    modWord="UNKNOWN";
                String modPos=sentence.pos(mod);
                if(first){
                    if(!fineGrainedPosCounts.get(leftWordPosDirNoAdj).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirNoAdj).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirNoAdj).put(modPos, fineGrainedPosCounts.get(leftWordPosDirNoAdj).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(leftWordPosDirNoAdj).containsKey(modWord))
                        fineGrainedWordCounts.get(leftWordPosDirNoAdj).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(leftWordPosDirNoAdj).put(modWord, fineGrainedWordCounts.get(leftWordPosDirNoAdj).get(modWord)+1);

                    wordPosDirAdjCount.put(leftWordPosDirNoAdj,wordPosDirAdjCount.get(leftWordPosDirNoAdj)+1);

                    if(!coarseGrainedPosCounts.get(leftPosDirNoAdj).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirNoAdj).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirNoAdj).put(modPos, coarseGrainedPosCounts.get(leftPosDirNoAdj).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(leftPosDirNoAdj).containsKey(modWord))
                        coarseGrainedWordCounts.get(leftPosDirNoAdj).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(leftPosDirNoAdj).put(modWord, coarseGrainedWordCounts.get(leftPosDirNoAdj).get(modWord)+1);

                    posDirAdjCount.put(leftPosDirNoAdj,posDirAdjCount.get(leftPosDirNoAdj)+1);
                }   else{
                    if(!fineGrainedPosCounts.get(leftWordPosDirAdj).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirAdj).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirAdj).put(modPos, fineGrainedPosCounts.get(leftWordPosDirAdj).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(leftWordPosDirAdj).containsKey(modWord))
                        fineGrainedWordCounts.get(leftWordPosDirAdj).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(leftWordPosDirAdj).put(modWord, fineGrainedWordCounts.get(leftWordPosDirAdj).get(modWord)+1);

                    wordPosDirAdjCount.put(leftWordPosDirNoAdj,wordPosDirAdjCount.get(leftWordPosDirNoAdj)+1);

                    if(!coarseGrainedPosCounts.get(leftPosDirAdj).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirAdj).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirAdj).put(modPos, coarseGrainedPosCounts.get(leftPosDirAdj).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(leftPosDirAdj).containsKey(modWord))
                        coarseGrainedWordCounts.get(leftPosDirAdj).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(leftPosDirAdj).put(modWord, coarseGrainedWordCounts.get(leftPosDirAdj).get(modWord)+1);


                    posDirAdjCount.put(leftPosDirNoAdj,posDirAdjCount.get(leftPosDirNoAdj)+1);
                }
                first=false;
            }

            // stop  after getting dependents on the left
            fineGrainedPosCounts.get(leftWordPosDirAdj).put(stop, fineGrainedPosCounts.get(leftWordPosDirAdj).get(stop)+1);
            wordPosDirAdjCount.put(leftWordPosDirAdj,wordPosDirAdjCount.get(leftWordPosDirAdj)+1);

            coarseGrainedPosCounts.get(leftPosDirAdj).put(stop,coarseGrainedPosCounts.get(leftPosDirAdj).get(stop)+1);
            posDirAdjCount.put(leftPosDirAdj,posDirAdjCount.get(leftPosDirAdj)+1);
        }


        // traversing right children
        if(revDepDic.get(m).snd.size()==0){
            // stop  without getting any dependents on the right
            fineGrainedPosCounts.get(rightWordPosDirNoAdj).put(stop, fineGrainedPosCounts.get(rightWordPosDirNoAdj).get(stop)+1);
            wordPosDirAdjCount.put(rightWordPosDirNoAdj,wordPosDirAdjCount.get(rightWordPosDirNoAdj)+1);

            coarseGrainedPosCounts.get(rightPosDirNoAdj).put(stop,coarseGrainedPosCounts.get(rightPosDirNoAdj).get(stop)+1);
            posDirAdjCount.put(rightPosDirNoAdj,posDirAdjCount.get(rightPosDirNoAdj)+1);
        }  else{
            boolean first=true;
            for(int mod:revDepDic.get(m).snd){
                traverseTree(mod,sentence,revDepDic);
                String modWord=sentence.word(mod);
                if(!wordCount.contains(modWord))
                    modWord="UNKNOWN";
                String modPos=sentence.pos(mod);
                if(first){
                    if(!fineGrainedPosCounts.get(rightWordPosDirNoAdj).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirNoAdj).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirNoAdj).put(modPos, fineGrainedPosCounts.get(rightWordPosDirNoAdj).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(rightWordPosDirNoAdj).containsKey(modWord))
                        fineGrainedWordCounts.get(rightWordPosDirNoAdj).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(rightWordPosDirNoAdj).put(modWord, fineGrainedWordCounts.get(rightWordPosDirNoAdj).get(modWord)+1);

                    wordPosDirAdjCount.put(rightWordPosDirNoAdj,wordPosDirAdjCount.get(rightWordPosDirNoAdj)+1);

                    if(!coarseGrainedPosCounts.get(rightPosDirNoAdj).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirNoAdj).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirNoAdj).put(modPos, coarseGrainedPosCounts.get(rightPosDirNoAdj).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(rightPosDirNoAdj).containsKey(modWord))
                        coarseGrainedWordCounts.get(rightPosDirNoAdj).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(rightPosDirNoAdj).put(modWord, coarseGrainedWordCounts.get(rightPosDirNoAdj).get(modWord)+1);

                    posDirAdjCount.put(rightPosDirNoAdj,posDirAdjCount.get(rightPosDirNoAdj)+1);
                }   else{
                    if(!fineGrainedPosCounts.get(rightWordPosDirAdj).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirAdj).put(modPos,1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirAdj).put(modPos, fineGrainedPosCounts.get(rightWordPosDirAdj).get(modPos)+1);

                    if(!fineGrainedWordCounts.get(rightWordPosDirAdj).containsKey(modWord))
                        fineGrainedWordCounts.get(rightWordPosDirAdj).put(modWord,1.0);
                    else
                        fineGrainedWordCounts.get(rightWordPosDirAdj).put(modWord, fineGrainedWordCounts.get(rightWordPosDirAdj).get(modWord)+1);

                    wordPosDirAdjCount.put(rightWordPosDirNoAdj,wordPosDirAdjCount.get(rightWordPosDirNoAdj)+1);

                    if(!coarseGrainedPosCounts.get(rightPosDirAdj).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirAdj).put(modPos,1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirAdj).put(modPos, coarseGrainedPosCounts.get(rightPosDirAdj).get(modPos)+1);

                    if(!coarseGrainedWordCounts.get(rightPosDirAdj).containsKey(modWord))
                        coarseGrainedWordCounts.get(rightPosDirAdj).put(modWord,1.0);
                    else
                        coarseGrainedWordCounts.get(rightPosDirAdj).put(modWord, coarseGrainedWordCounts.get(rightPosDirAdj).get(modWord)+1);


                    posDirAdjCount.put(rightPosDirNoAdj,posDirAdjCount.get(rightPosDirNoAdj)+1);
                }
                first=false;
            }

            // stop  after getting dependents on the right
            fineGrainedPosCounts.get(rightWordPosDirAdj).put(stop, fineGrainedPosCounts.get(rightWordPosDirAdj).get(stop)+1);
            wordPosDirAdjCount.put(rightWordPosDirAdj,wordPosDirAdjCount.get(rightWordPosDirAdj)+1);

            coarseGrainedPosCounts.get(rightPosDirAdj).put(stop,coarseGrainedPosCounts.get(rightPosDirAdj).get(stop)+1);
            posDirAdjCount.put(rightPosDirAdj,posDirAdjCount.get(rightPosDirAdj)+1);
        }

    }

}
