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
    Pattern numPat;
    int min = 6;
    HashSet<String> wordCount;
    HashSet<String> posList;
    double wordSmoothing;
    double posSmoothing;
    /**
     * The key is the tuple <pos+word+dir+val>
     * The second key is the pos
     */
    HashMap<String, HashMap<String, Double>> fineGrainedPosCounts;
    HashMap<String, Integer> wordPosDirValCount;
    /**
     * The key is the tuple <pos+dir+val>
     * The second key is the pos
     */
    HashMap<String, HashMap<String, Double>> coarseGrainedPosCounts;
    HashMap<String, Integer> posDirValCount;
    HashMap<String, Integer> word2PosDirValCount;
    HashMap<String, Integer> pos2DirValCount;
    /**
     * The key is the tuple <pos+word+dir+val>
     * The second key is the word
     */
    HashMap<String, HashMap<String, Double>> fineGrainedWordCounts;
    /**
     * The key is the tuple <pos+dir+val>
     * The second key is the word
     */
    HashMap<String, HashMap<String, Double>> coarseGrainedWordCounts;
    HashMap<String, HashMap<String, Double>> wordPosCount;
    HashMap<String, Integer> posCount;
    HashMap<String, HashMap<String, Double>> posPosCount;

    public GenerativeModel(double wordSmoothing, double posSmoothing) {
        wordCount = new HashSet<String>();
        posList = new HashSet<String>();
        fineGrainedPosCounts = new HashMap<String, HashMap<String, Double>>();
        coarseGrainedPosCounts = new HashMap<String, HashMap<String, Double>>();
        fineGrainedWordCounts = new HashMap<String, HashMap<String, Double>>();
        coarseGrainedWordCounts = new HashMap<String, HashMap<String, Double>>();
        wordPosDirValCount = new HashMap<String, Integer>();
        posDirValCount = new HashMap<String, Integer>();
        this.wordSmoothing = wordSmoothing;
        this.posSmoothing = posSmoothing;
        wordPosCount = new HashMap<String, HashMap<String, Double>>();
        posCount = new HashMap<String, Integer>();
        posPosCount = new HashMap<String, HashMap<String, Double>>();
        word2PosDirValCount = new HashMap<String, Integer>();
        pos2DirValCount = new HashMap<String, Integer>();
        numPat = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");
        initializePuncs();
    }

    private static void initializePuncs() {
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

    public static void saveModel(GenerativeModel generativeModel, String modelPath) throws Exception {
        ObjectOutput writer = new ObjectOutputStream(new FileOutputStream(modelPath));
        writer.writeObject(generativeModel);
        writer.flush();
        writer.close();
    }

    public static GenerativeModel loadModel(String modelPath) throws Exception {
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(modelPath));
        GenerativeModel gm = (GenerativeModel) reader.readObject();
        return gm;
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
                String word = sentence.word(i);//.toLowerCase();
                Matcher matcher = numPat.matcher(word);
                if (matcher.matches())
                    word = "<num>";

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
        String direction = (head > mod) ? "l" : "r";
        String valency = val ? "v" : "!v";

        String hw = sentence.word(head);//.toLowerCase();
        Matcher matcher = numPat.matcher(hw);
        if (matcher.matches())
            hw = "<num>";

        String hp = sentence.pos(head);
        if (!wordCount.contains(hw))
            hw = "UNKNOWN";

        String mp = "STOP";
        String mw = "STOP";
        if (mod >= 0 && mod < sentence.length()) {
            mw = sentence.word(mod);//.toLowerCase();
            matcher = numPat.matcher(mw);
            if (matcher.matches())
                mw = "<num>";
            mp = sentence.pos(mod);
            if (!wordCount.contains(mw))
                mw = "UNKNOWN";
        }

        String wordPosDirVal = hw + "|" + hp + "|" + direction + "|" + valency;
        String posDirVal = hp + "|" + direction + "|" + valency;

        String word2PosDirVal = mp + "|" + hw + "|" + hp + "|" + direction + "|" + valency;
        String pos2DirVal = mp + "|" + hp + "|" + direction + "|" + valency;

        double f1 = 0;// wordSmoothing*wordCount.size();
        double n1 = wordSmoothing;
        double n2 = posSmoothing;
        double f2 = 0;//posSmoothing*posList.size();

        double n3 = wordSmoothing;
        double n4 = posSmoothing;
        double f3 = 0;//wordSmoothing*wordCount.size();
        double f4 = 0;//posSmoothing*posList.size();

        double n5 = 0;
        double n6 = 0;
        int f5 = 0;


        if (posCount.containsKey(hp)) {
            f5 = posCount.get(hp);

            if (wordPosCount.containsKey(hp) &&
                    wordPosCount.get(hp).containsKey(mw)) {
                n5 = wordPosCount.get(hp).get(mw);
            }

            if (posPosCount.containsKey(hp) && posPosCount.get(hp).containsKey(mp)) {
                n6 = posPosCount.get(hp).get(mp);
            }
        }


        if (word2PosDirValCount.containsKey(word2PosDirVal)) {
            f1 += word2PosDirValCount.get(word2PosDirVal);
            n1 = wordSmoothing;

            if (fineGrainedWordCounts.get(word2PosDirVal).containsKey(mw))
                n1 += fineGrainedWordCounts.get(word2PosDirVal).get(mw);
        }

        if (pos2DirValCount.containsKey(pos2DirVal)) {
            f2 += pos2DirValCount.get(pos2DirVal);
            n2 = wordSmoothing;

            if (coarseGrainedWordCounts.get(pos2DirVal).containsKey(mw))
                n2 += coarseGrainedWordCounts.get(pos2DirVal).get(mw);
        }

        if (wordPosDirValCount.containsKey(wordPosDirVal)) {
            f3 += wordPosDirValCount.get(wordPosDirVal);
            n3 = posSmoothing;

            if (fineGrainedPosCounts.get(wordPosDirVal).containsKey(mp))
                n3 += fineGrainedPosCounts.get(wordPosDirVal).get(mp);
        }


        if (posDirValCount.containsKey(posDirVal)) {
            f4 += posDirValCount.get(posDirVal);
            n4 = posSmoothing;

            if (coarseGrainedPosCounts.get(posDirVal).containsKey(mp))
                n4 += coarseGrainedPosCounts.get(posDirVal).get(mp);
        }


        int u1 = 0;
        if (fineGrainedWordCounts.containsKey(word2PosDirVal))
            u1 = fineGrainedWordCounts.get(word2PosDirVal).size();
        // word2PosDirValCount.size();
        double l1 = f1 / (f1 + 1 * u1);
        if (Double.isNaN(l1) || Double.isInfinite(l1))
            l1 = 0;

        int u2 = 0;
        if (coarseGrainedWordCounts.containsKey(pos2DirVal))
            u2 = coarseGrainedWordCounts.get(pos2DirVal).size();
        //pos2DirValCount.size();
        double l2 = f2 / (f2 + 5 * u2);
        if (Double.isNaN(l2) || Double.isInfinite(l2))
            l2 = 0;

        int u3 = 0;
        if (fineGrainedPosCounts.containsKey(wordPosDirVal))
            u3 = fineGrainedPosCounts.get(wordPosDirVal).size();
        //wordPosDirValCount.size();
        double l3 = f3 / (f3 + 1 * u3);
        if (Double.isNaN(l3) || Double.isInfinite(l3))
            l3 = 0;


        int u4 = 0;
        if (coarseGrainedPosCounts.containsKey(posDirVal))
            u4 = coarseGrainedPosCounts.get(posDirVal).size();
        // posDirValCount.size();
        double l4 = f4 / (f4 + 5 * u4);
        if (Double.isNaN(l4) || Double.isInfinite(l4))
            l4 = 0;

        double fact1 = n1 / (f1+ wordSmoothing*wordCount.size());
        double fact2 = n2 / (f2+ posSmoothing*posCount.size());
        double fact3 = n5 / f5;

        double fact4 = n3 /(f3+ wordSmoothing*wordCount.size());
        double fact5 = n4 / (f4+ posSmoothing*posCount.size());
        double fact6 = n6 / f5;

        if (Double.isNaN(fact1))
            fact1 = 0;
        if (Double.isNaN(fact2))
            fact2 = 0;
        if (Double.isNaN(fact3))
            fact3 = 0;
        if (Double.isNaN(fact4))
            fact4 = 0;
        if (Double.isNaN(fact5))
            fact5 = 0;
        if (Double.isNaN(fact6))
            fact6 = 0;

        double p = Math.log(l1 * fact1 + (1.0 - l1) * (l2 * fact2 + (1.0 - l2) * fact3)) + Math.log(l3 * fact4 + (1 - l3) * (l4 * fact5 + (1. - l4) * fact6));

        if (mw.equals("STOP"))
            p = Math.log(l3 * fact4 + (1 - l3) * (l4 * fact5 + (1. - l4) * fact6));

        if (Double.isNaN(p))
            p = Double.NEGATIVE_INFINITY;
        if (Double.isInfinite(p))
            p = Double.NEGATIVE_INFINITY;
        return p;
    }

    public void parse(ArrayList<Sentence> sentences) {
        System.err.print("parsing the sentences...");
        GraphBasedParser parser = new GraphBasedParser(this);
        int allDeps = 0;
        double correctDeps = 0.0;

        for (int i = 0; i < sentences.size(); i++) {
            Sentence gSentence = sentences.get(i);
            Sentence pSentence = parser.eisner2ndOrder(gSentence);

            for (int j = 1; j < pSentence.length(); j++) {
                if (!punctuations.contains(pSentence.pos(j))) {
                    allDeps++;
                    if (pSentence.head(j) == gSentence.head(j))
                        correctDeps++;
                }
            }

            if ((i + 1) % 100 == 0)
                System.err.print((i + 1) + "...");
        }
        System.err.print("\n");

        double accuracy = 100.0 * correctDeps / allDeps;
        System.err.println("accuracy:\t" + accuracy);

    }

    private void traverseTree(int m, Sentence sentence, HashMap<Integer, Pair<TreeSet<Integer>, TreeSet<Integer>>> revDepDic) {
        String word = sentence.word(m);//.toLowerCase();
        Matcher matcher = numPat.matcher(word);
        if (matcher.matches())
            word = "<num>";
        String pos = sentence.pos(m);

        if (!wordCount.contains(word))
            word = "UNKNOWN";

        String stop = "STOP";

        String leftWordPosDirNoVal = word + "|" + pos + "|l|!v";
        String rightWordPosDirNoVal = word + "|" + pos + "|r|!v";
        String leftPosDirNoVal = pos + "|l|!v";
        String rightPosDirNoVal = pos + "|r|!v";

        String leftWordPosDirVal = word + "|" + pos + "|l|v";
        String rightWordPosDirVal = word + "|" + pos + "|r|v";
        String leftPosDirVal = pos + "|l|v";
        String rightPosDirVal = pos + "|r|v";

        if (!posPosCount.containsKey(pos)) {
            posPosCount.put(pos, new HashMap<String, Double>());
            if (m != 0)
                posPosCount.get(pos).put(stop, 0.0);
            posCount.put(pos, 0);
        }
        if (m != 0) {
            posPosCount.get(pos).put(stop, posPosCount.get(pos).get(stop) + 2);
            posCount.put(pos, posCount.get(pos) + 2);
        }

        //todo
        // traversing left children
        if (revDepDic.get(m).fst.size() == 0) {
            if (m != 0) {
                // stop  without getting any dependents on the left
                //System.err.println(m+"->l(!a)->stop");
                if (!fineGrainedPosCounts.containsKey(leftWordPosDirNoVal))
                    fineGrainedPosCounts.put(leftWordPosDirNoVal, new HashMap<String, Double>());
                if (!fineGrainedPosCounts.get(leftWordPosDirNoVal).containsKey(stop))
                    fineGrainedPosCounts.get(leftWordPosDirNoVal).put(stop, 1.);
                else
                    fineGrainedPosCounts.get(leftWordPosDirNoVal).put(stop, fineGrainedPosCounts.get(leftWordPosDirNoVal).get(stop) + 1);

                if (!wordPosDirValCount.containsKey(leftWordPosDirNoVal))
                    wordPosDirValCount.put(leftWordPosDirNoVal, 1);
                else
                    wordPosDirValCount.put(leftWordPosDirNoVal, wordPosDirValCount.get(leftWordPosDirNoVal) + 1);

                if (!coarseGrainedPosCounts.containsKey(leftPosDirNoVal))
                    coarseGrainedPosCounts.put(leftPosDirNoVal, new HashMap<String, Double>());
                if (!coarseGrainedPosCounts.get(leftPosDirNoVal).containsKey(stop))
                    coarseGrainedPosCounts.get(leftPosDirNoVal).put(stop, 1.);
                else
                    coarseGrainedPosCounts.get(leftPosDirNoVal).put(stop, coarseGrainedPosCounts.get(leftPosDirNoVal).get(stop) + 1);

                if (!posDirValCount.containsKey(leftPosDirNoVal))
                    posDirValCount.put(leftPosDirNoVal, 1);
                else
                    posDirValCount.put(leftPosDirNoVal, posDirValCount.get(leftPosDirNoVal) + 1);
            }
        } else {
            boolean first = true;
            for (int mod : revDepDic.get(m).fst.descendingSet()) {
                traverseTree(mod, sentence, revDepDic);
                String modWord = sentence.word(mod);//.toLowerCase();
                matcher = numPat.matcher(modWord);
                if (matcher.matches())
                    modWord = "<num>";
                if (!wordCount.contains(modWord))
                    modWord = "UNKNOWN";
                String modPos = sentence.pos(mod);

                posCount.put(pos, posCount.get(pos) + 1);
                if (posPosCount.get(pos).containsKey(modPos))
                    posPosCount.get(pos).put(modPos, posPosCount.get(pos).get(modPos) + 1);
                else
                    posPosCount.get(pos).put(modPos, 1.);

                if (!wordPosCount.containsKey(pos))
                    wordPosCount.put(pos, new HashMap<String, Double>());
                if (wordPosCount.get(pos).containsKey(modWord))
                    wordPosCount.get(pos).put(modWord, wordPosCount.get(pos).get(modWord) + 1);
                else
                    wordPosCount.get(pos).put(modWord, 1.);


                if (first) {
                    //System.err.println(m+"->l(!a)->"+mod);
                    String left2WordPosDirNoVal = modPos + "|" + word + "|" + pos + "|l|!v";
                    String left2PosDirNoVal = modPos + "|" + pos + "|l|!v";
                    if (!fineGrainedPosCounts.containsKey(leftWordPosDirNoVal))
                        fineGrainedPosCounts.put(leftWordPosDirNoVal, new HashMap<String, Double>());
                    if (!fineGrainedPosCounts.get(leftWordPosDirNoVal).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirNoVal).put(modPos, 1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirNoVal).put(modPos, fineGrainedPosCounts.get(leftWordPosDirNoVal).get(modPos) + 1);

                    if (!fineGrainedWordCounts.containsKey(left2WordPosDirNoVal))
                        fineGrainedWordCounts.put(left2WordPosDirNoVal, new HashMap<String, Double>());
                    if (!fineGrainedWordCounts.get(left2WordPosDirNoVal).containsKey(modWord))
                        fineGrainedWordCounts.get(left2WordPosDirNoVal).put(modWord, 1.0);
                    else
                        fineGrainedWordCounts.get(left2WordPosDirNoVal).put(modWord, fineGrainedWordCounts.get(left2WordPosDirNoVal).get(modWord) + 1);

                    if (!coarseGrainedPosCounts.containsKey(leftPosDirNoVal))
                        coarseGrainedPosCounts.put(leftPosDirNoVal, new HashMap<String, Double>());
                    if (!coarseGrainedPosCounts.get(leftPosDirNoVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirNoVal).put(modPos, 1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirNoVal).put(modPos, coarseGrainedPosCounts.get(leftPosDirNoVal).get(modPos) + 1);

                    if (!coarseGrainedWordCounts.containsKey(left2PosDirNoVal))
                        coarseGrainedWordCounts.put(left2PosDirNoVal, new HashMap<String, Double>());
                    if (!coarseGrainedWordCounts.get(left2PosDirNoVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(left2PosDirNoVal).put(modWord, 1.0);
                    else
                        coarseGrainedWordCounts.get(left2PosDirNoVal).put(modWord, coarseGrainedWordCounts.get(left2PosDirNoVal).get(modWord) + 1);

                    if (!wordPosDirValCount.containsKey(leftWordPosDirNoVal))
                        wordPosDirValCount.put(leftWordPosDirNoVal, 1);
                    else
                        wordPosDirValCount.put(leftWordPosDirNoVal, wordPosDirValCount.get(leftWordPosDirNoVal) + 1);

                    if (!posDirValCount.containsKey(leftPosDirNoVal))
                        posDirValCount.put(leftPosDirNoVal, 1);
                    else
                        posDirValCount.put(leftPosDirNoVal, posDirValCount.get(leftPosDirNoVal) + 1);

                    if (!word2PosDirValCount.containsKey(left2WordPosDirNoVal))
                        word2PosDirValCount.put(left2WordPosDirNoVal, 1);
                    else
                        word2PosDirValCount.put(left2WordPosDirNoVal, word2PosDirValCount.get(left2WordPosDirNoVal) + 1);

                    if (!pos2DirValCount.containsKey(left2PosDirNoVal))
                        pos2DirValCount.put(left2PosDirNoVal, 1);
                    else
                        pos2DirValCount.put(left2PosDirNoVal, pos2DirValCount.get(left2PosDirNoVal) + 1);
                } else {
                    //System.err.println(m+"->l(a)->"+mod);
                    String left2WordPosDirVal = modPos + "|" + word + "|" + pos + "|l|v";
                    String left2PosDirVal = modPos + "|" + pos + "|l|v";

                    if (!fineGrainedPosCounts.containsKey(leftWordPosDirVal))
                        fineGrainedPosCounts.put(leftWordPosDirVal, new HashMap<String, Double>());
                    if (!fineGrainedPosCounts.get(leftWordPosDirVal).containsKey(modPos))
                        fineGrainedPosCounts.get(leftWordPosDirVal).put(modPos, 1.0);
                    else
                        fineGrainedPosCounts.get(leftWordPosDirVal).put(modPos, fineGrainedPosCounts.get(leftWordPosDirVal).get(modPos) + 1);

                    if (!fineGrainedWordCounts.containsKey(left2WordPosDirVal))
                        fineGrainedWordCounts.put(left2WordPosDirVal, new HashMap<String, Double>());
                    if (!fineGrainedWordCounts.get(left2WordPosDirVal).containsKey(modWord))
                        fineGrainedWordCounts.get(left2WordPosDirVal).put(modWord, 1.0);
                    else
                        fineGrainedWordCounts.get(left2WordPosDirVal).put(modWord, fineGrainedWordCounts.get(left2WordPosDirVal).get(modWord) + 1);

                    if (!coarseGrainedPosCounts.containsKey(leftPosDirVal))
                        coarseGrainedPosCounts.put(leftPosDirVal, new HashMap<String, Double>());
                    if (!coarseGrainedPosCounts.get(leftPosDirVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(leftPosDirVal).put(modPos, 1.0);
                    else
                        coarseGrainedPosCounts.get(leftPosDirVal).put(modPos, coarseGrainedPosCounts.get(leftPosDirVal).get(modPos) + 1);

                    if (!coarseGrainedWordCounts.containsKey(left2PosDirVal))
                        coarseGrainedWordCounts.put(left2PosDirVal, new HashMap<String, Double>());
                    if (!coarseGrainedWordCounts.get(left2PosDirVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(left2PosDirVal).put(modWord, 1.0);
                    else
                        coarseGrainedWordCounts.get(left2PosDirVal).put(modWord, coarseGrainedWordCounts.get(left2PosDirVal).get(modWord) + 1);

                    if (!wordPosDirValCount.containsKey(leftWordPosDirVal))
                        wordPosDirValCount.put(leftWordPosDirVal, 1);
                    else
                        wordPosDirValCount.put(leftWordPosDirVal, wordPosDirValCount.get(leftWordPosDirVal) + 1);

                    if (!posDirValCount.containsKey(leftPosDirVal))
                        posDirValCount.put(leftPosDirVal, 1);
                    else
                        posDirValCount.put(leftPosDirVal, posDirValCount.get(leftPosDirVal) + 1);

                    if (!word2PosDirValCount.containsKey(left2WordPosDirVal))
                        word2PosDirValCount.put(left2WordPosDirVal, 1);
                    else
                        word2PosDirValCount.put(left2WordPosDirVal, word2PosDirValCount.get(left2WordPosDirVal) + 1);

                    if (!pos2DirValCount.containsKey(left2PosDirVal))
                        pos2DirValCount.put(left2PosDirVal, 1);
                    else
                        pos2DirValCount.put(left2PosDirVal, pos2DirValCount.get(left2PosDirVal) + 1);
                }
                first = false;
            }
            if (m != 0) {
                // stop  after getting dependents on the left
                if (!fineGrainedPosCounts.containsKey(leftWordPosDirVal))
                    fineGrainedPosCounts.put(leftWordPosDirVal, new HashMap<String, Double>());
                if (!fineGrainedPosCounts.get(leftWordPosDirVal).containsKey(stop))
                    fineGrainedPosCounts.get(leftWordPosDirVal).put(stop, 1.);
                else
                    fineGrainedPosCounts.get(leftWordPosDirVal).put(stop, fineGrainedPosCounts.get(leftWordPosDirVal).get(stop) + 1);

                if (!wordPosDirValCount.containsKey(leftWordPosDirVal))
                    wordPosDirValCount.put(leftWordPosDirVal, 1);
                else
                    wordPosDirValCount.put(leftWordPosDirVal, wordPosDirValCount.get(leftWordPosDirVal) + 1);
                //System.err.println(m+"->l(a)->stop");

                if (!coarseGrainedPosCounts.containsKey(leftPosDirVal))
                    coarseGrainedPosCounts.put(leftPosDirVal, new HashMap<String, Double>());
                if (!coarseGrainedPosCounts.get(leftPosDirVal).containsKey(stop))
                    coarseGrainedPosCounts.get(leftPosDirVal).put(stop, 1.);
                else
                    coarseGrainedPosCounts.get(leftPosDirVal).put(stop, coarseGrainedPosCounts.get(leftPosDirVal).get(stop) + 1);

                if (!posDirValCount.containsKey(leftPosDirVal))
                    posDirValCount.put(leftPosDirVal, 1);
                else
                    posDirValCount.put(leftPosDirVal, posDirValCount.get(leftPosDirVal) + 1);
            }
        }

        //todo
        // traversing right children
        if (revDepDic.get(m).snd.size() == 0) {
            if (m != 0) {
                // stop  without getting any dependents on the right
                //System.err.println(m+"->l(!a)->stop");
                if (!fineGrainedPosCounts.containsKey(rightWordPosDirNoVal))
                    fineGrainedPosCounts.put(rightWordPosDirNoVal, new HashMap<String, Double>());
                if (!fineGrainedPosCounts.get(rightWordPosDirNoVal).containsKey(stop))
                    fineGrainedPosCounts.get(rightWordPosDirNoVal).put(stop, 1.);
                else
                    fineGrainedPosCounts.get(rightWordPosDirNoVal).put(stop, fineGrainedPosCounts.get(rightWordPosDirNoVal).get(stop) + 1);

                if (!wordPosDirValCount.containsKey(rightWordPosDirNoVal))
                    wordPosDirValCount.put(rightWordPosDirNoVal, 1);
                else
                    wordPosDirValCount.put(rightWordPosDirNoVal, wordPosDirValCount.get(rightWordPosDirNoVal) + 1);

                if (!coarseGrainedPosCounts.containsKey(rightPosDirNoVal))
                    coarseGrainedPosCounts.put(rightPosDirNoVal, new HashMap<String, Double>());
                if (!coarseGrainedPosCounts.get(rightPosDirNoVal).containsKey(stop))
                    coarseGrainedPosCounts.get(rightPosDirNoVal).put(stop, 1.);
                else
                    coarseGrainedPosCounts.get(rightPosDirNoVal).put(stop, coarseGrainedPosCounts.get(rightPosDirNoVal).get(stop) + 1);

                if (!posDirValCount.containsKey(rightPosDirNoVal))
                    posDirValCount.put(rightPosDirNoVal, 1);
                else
                    posDirValCount.put(rightPosDirNoVal, posDirValCount.get(rightPosDirNoVal) + 1);
            }
        } else {
            boolean first = true;
            for (int mod : revDepDic.get(m).snd) {
                traverseTree(mod, sentence, revDepDic);
                String modWord = sentence.word(mod);//.toLowerCase();
                matcher = numPat.matcher(modWord);
                if (matcher.matches())
                    modWord = "<num>";
                if (!wordCount.contains(modWord))
                    modWord = "UNKNOWN";
                String modPos = sentence.pos(mod);

                posCount.put(pos, posCount.get(pos) + 1);
                if (posPosCount.get(pos).containsKey(modPos))
                    posPosCount.get(pos).put(modPos, posPosCount.get(pos).get(modPos) + 1);
                else
                    posPosCount.get(pos).put(modPos, 1.);

                if (!wordPosCount.containsKey(pos))
                    wordPosCount.put(pos, new HashMap<String, Double>());
                if (wordPosCount.get(pos).containsKey(modWord))
                    wordPosCount.get(pos).put(modWord, wordPosCount.get(pos).get(modWord) + 1);
                else
                    wordPosCount.get(pos).put(modWord, 1.);


                if (first) {
                    //System.err.println(m+"->l(!a)->"+mod);
                    String right2WordPosDirNoVal = modPos + "|" + word + "|" + pos + "|r|!v";
                    String right2PosDirNoVal = modPos + "|" + pos + "|r|!v";
                    if (!fineGrainedPosCounts.containsKey(rightWordPosDirNoVal))
                        fineGrainedPosCounts.put(rightWordPosDirNoVal, new HashMap<String, Double>());
                    if (!fineGrainedPosCounts.get(rightWordPosDirNoVal).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirNoVal).put(modPos, 1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirNoVal).put(modPos, fineGrainedPosCounts.get(rightWordPosDirNoVal).get(modPos) + 1);

                    if (!fineGrainedWordCounts.containsKey(right2WordPosDirNoVal))
                        fineGrainedWordCounts.put(right2WordPosDirNoVal, new HashMap<String, Double>());
                    if (!fineGrainedWordCounts.get(right2WordPosDirNoVal).containsKey(modWord))
                        fineGrainedWordCounts.get(right2WordPosDirNoVal).put(modWord, 1.0);
                    else
                        fineGrainedWordCounts.get(right2WordPosDirNoVal).put(modWord, fineGrainedWordCounts.get(right2WordPosDirNoVal).get(modWord) + 1);

                    if (!coarseGrainedPosCounts.containsKey(rightPosDirNoVal))
                        coarseGrainedPosCounts.put(rightPosDirNoVal, new HashMap<String, Double>());
                    if (!coarseGrainedPosCounts.get(rightPosDirNoVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirNoVal).put(modPos, 1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirNoVal).put(modPos, coarseGrainedPosCounts.get(rightPosDirNoVal).get(modPos) + 1);

                    if (!coarseGrainedWordCounts.containsKey(right2PosDirNoVal))
                        coarseGrainedWordCounts.put(right2PosDirNoVal, new HashMap<String, Double>());
                    if (!coarseGrainedWordCounts.get(right2PosDirNoVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(right2PosDirNoVal).put(modWord, 1.0);
                    else
                        coarseGrainedWordCounts.get(right2PosDirNoVal).put(modWord, coarseGrainedWordCounts.get(right2PosDirNoVal).get(modWord) + 1);

                    if (!wordPosDirValCount.containsKey(rightWordPosDirNoVal))
                        wordPosDirValCount.put(rightWordPosDirNoVal, 1);
                    else
                        wordPosDirValCount.put(rightWordPosDirNoVal, wordPosDirValCount.get(rightWordPosDirNoVal) + 1);

                    if (!posDirValCount.containsKey(rightPosDirNoVal))
                        posDirValCount.put(rightPosDirNoVal, 1);
                    else
                        posDirValCount.put(rightPosDirNoVal, posDirValCount.get(rightPosDirNoVal) + 1);

                    if (!word2PosDirValCount.containsKey(right2WordPosDirNoVal))
                        word2PosDirValCount.put(right2WordPosDirNoVal, 1);
                    else
                        word2PosDirValCount.put(right2WordPosDirNoVal, word2PosDirValCount.get(right2WordPosDirNoVal) + 1);

                    if (!pos2DirValCount.containsKey(right2PosDirNoVal))
                        pos2DirValCount.put(right2PosDirNoVal, 1);
                    else
                        pos2DirValCount.put(right2PosDirNoVal, pos2DirValCount.get(right2PosDirNoVal) + 1);
                } else {
                    //System.err.println(m+"->l(a)->"+mod);
                    String right2WordPosDirVal = modPos + "|" + word + "|" + pos + "|r|v";
                    String right2PosDirVal = modPos + "|" + pos + "|r|v";

                    if (!fineGrainedPosCounts.containsKey(rightWordPosDirVal))
                        fineGrainedPosCounts.put(rightWordPosDirVal, new HashMap<String, Double>());
                    if (!fineGrainedPosCounts.get(rightWordPosDirVal).containsKey(modPos))
                        fineGrainedPosCounts.get(rightWordPosDirVal).put(modPos, 1.0);
                    else
                        fineGrainedPosCounts.get(rightWordPosDirVal).put(modPos, fineGrainedPosCounts.get(rightWordPosDirVal).get(modPos) + 1);

                    if (!fineGrainedWordCounts.containsKey(right2WordPosDirVal))
                        fineGrainedWordCounts.put(right2WordPosDirVal, new HashMap<String, Double>());
                    if (!fineGrainedWordCounts.get(right2WordPosDirVal).containsKey(modWord))
                        fineGrainedWordCounts.get(right2WordPosDirVal).put(modWord, 1.0);
                    else
                        fineGrainedWordCounts.get(right2WordPosDirVal).put(modWord, fineGrainedWordCounts.get(right2WordPosDirVal).get(modWord) + 1);

                    if (!coarseGrainedPosCounts.containsKey(rightPosDirVal))
                        coarseGrainedPosCounts.put(rightPosDirVal, new HashMap<String, Double>());
                    if (!coarseGrainedPosCounts.get(rightPosDirVal).containsKey(modPos))
                        coarseGrainedPosCounts.get(rightPosDirVal).put(modPos, 1.0);
                    else
                        coarseGrainedPosCounts.get(rightPosDirVal).put(modPos, coarseGrainedPosCounts.get(rightPosDirVal).get(modPos) + 1);

                    if (!coarseGrainedWordCounts.containsKey(right2PosDirVal))
                        coarseGrainedWordCounts.put(right2PosDirVal, new HashMap<String, Double>());
                    if (!coarseGrainedWordCounts.get(right2PosDirVal).containsKey(modWord))
                        coarseGrainedWordCounts.get(right2PosDirVal).put(modWord, 1.0);
                    else
                        coarseGrainedWordCounts.get(right2PosDirVal).put(modWord, coarseGrainedWordCounts.get(right2PosDirVal).get(modWord) + 1);

                    if (!wordPosDirValCount.containsKey(rightWordPosDirVal))
                        wordPosDirValCount.put(rightWordPosDirVal, 1);
                    else
                        wordPosDirValCount.put(rightWordPosDirVal, wordPosDirValCount.get(rightWordPosDirVal) + 1);

                    if (!posDirValCount.containsKey(rightPosDirVal))
                        posDirValCount.put(rightPosDirVal, 1);
                    else
                        posDirValCount.put(rightPosDirVal, posDirValCount.get(rightPosDirVal) + 1);

                    if (!word2PosDirValCount.containsKey(right2WordPosDirVal))
                        word2PosDirValCount.put(right2WordPosDirVal, 1);
                    else
                        word2PosDirValCount.put(right2WordPosDirVal, word2PosDirValCount.get(right2WordPosDirVal) + 1);

                    if (!pos2DirValCount.containsKey(right2PosDirVal))
                        pos2DirValCount.put(right2PosDirVal, 1);
                    else
                        pos2DirValCount.put(right2PosDirVal, pos2DirValCount.get(right2PosDirVal) + 1);
                }
                first = false;
            }
            if (m != 0) {
                // stop  after getting dependents on the right
                if (!fineGrainedPosCounts.containsKey(rightWordPosDirVal))
                    fineGrainedPosCounts.put(rightWordPosDirVal, new HashMap<String, Double>());
                if (!fineGrainedPosCounts.get(rightWordPosDirVal).containsKey(stop))
                    fineGrainedPosCounts.get(rightWordPosDirVal).put(stop, 1.);
                else
                    fineGrainedPosCounts.get(rightWordPosDirVal).put(stop, fineGrainedPosCounts.get(rightWordPosDirVal).get(stop) + 1);

                if (!wordPosDirValCount.containsKey(rightWordPosDirVal))
                    wordPosDirValCount.put(rightWordPosDirVal, 1);
                else
                    wordPosDirValCount.put(rightWordPosDirVal, wordPosDirValCount.get(rightWordPosDirVal) + 1);
                //System.err.println(m+"->l(a)->stop");

                if (!coarseGrainedPosCounts.containsKey(rightPosDirVal))
                    coarseGrainedPosCounts.put(rightPosDirVal, new HashMap<String, Double>());
                if (!coarseGrainedPosCounts.get(rightPosDirVal).containsKey(stop))
                    coarseGrainedPosCounts.get(rightPosDirVal).put(stop, 1.);
                else
                    coarseGrainedPosCounts.get(rightPosDirVal).put(stop, coarseGrainedPosCounts.get(rightPosDirVal).get(stop) + 1);

                if (!posDirValCount.containsKey(rightPosDirVal))
                    posDirValCount.put(rightPosDirVal, 1);
                else
                    posDirValCount.put(rightPosDirVal, posDirValCount.get(rightPosDirVal) + 1);
            }
        }

    }

}
