package Trainer;

import Accessories.MSTReader;
import Classifier.OnlineClassifier;
import Decoder.FeatureExtractor;
import Decoder.GraphBasedParser;
import Structures.Sentence;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 11:28 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class PartialTreeTrainer {
    static HashSet<String> punctuations = new HashSet<String>();

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

    public static void train(ArrayList<Sentence> trainSentences, ArrayList<Sentence> devSentences, ArrayList<String> possibleLabels,
                             OnlineClassifier onlineClassifier, String modelPath, int maxIter, boolean trainStructuredForFullTrees, String outPath, boolean useHandCraftedRules, boolean softConstraint) throws Exception {

        initializePuncs();

        for (int iter = 0; iter < maxIter; iter++) {
            int numDep = 0;
            double correct = 0;
            GraphBasedParser trainParser = new GraphBasedParser(onlineClassifier, possibleLabels);

            System.out.println("*********************************************************");
            System.out.println("iteration: " + iter);
            int senCount = 0;
            for (Sentence sentence : trainSentences) {
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }
                boolean isCompleteTree = true;
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (!sentence.hasHead(ch)) {
                        isCompleteTree = false;
                        break;
                    }
                }
                if (!isCompleteTree || !trainStructuredForFullTrees) {
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        if (sentence.hasHead(ch)) {
                            numDep++;
                            // finding the best head
                            int goldHead = sentence.head(ch);
                            String goldLabel = sentence.label(ch);

                            int argmax = 0;
                            String bestLabel = "";
                            double max = Double.NEGATIVE_INFINITY;

                            for (String label : possibleLabels) {
                                for (int h = 0; h < sentence.length(); h++) {
                                    double score = onlineClassifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, h, ch), false);
                                    if (score > max) {
                                        max = score;
                                        bestLabel = label;
                                        argmax = h;
                                    }
                                }
                            }

                            if (argmax != goldHead || bestLabel.equals(goldLabel)) {
                                ArrayList<String> predictedFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, argmax, ch);
                                ArrayList<String> goldFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, goldHead, ch);

                                for (String predicted : predictedFeatures)
                                    onlineClassifier.updateWeight(predicted, -1);
                                for (String gold : goldFeatures)
                                    onlineClassifier.updateWeight(gold, 1);
                            } else {
                                correct++;
                            }
                        }
                    }
                    onlineClassifier.incrementIteration();
                } else {
                    Sentence parseTree = trainParser.eisner1stOrder(sentence, false);

                    for (int ch = 1; ch < sentence.length(); ch++) {
                        numDep++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);
                        if (possibleLabels.size() <= 1)
                            goldLabel = "";

                        int argmax = parseTree.head(ch);
                        String bestLabel = parseTree.label(ch);

                        if (argmax != goldHead || (possibleLabels.size() > 1 && !bestLabel.equals(goldLabel))) {
                            ArrayList<String> predictedFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, argmax, ch);
                            ArrayList<String> goldFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, goldHead, ch);

                            for (String predicted : predictedFeatures)
                                onlineClassifier.updateWeight(predicted, -1);
                            for (String gold : goldFeatures)
                                onlineClassifier.updateWeight(gold, 1);
                        } else {
                            correct++;
                        }
                    }
                    onlineClassifier.incrementIteration();

                }
            }
            System.out.println("");
            double accuracy = 100.0 * correct / numDep;
            System.out.println("size : " + onlineClassifier.size());
            System.out.println("accuracy : " + accuracy);

            System.out.print("\nsaving current model...");
            onlineClassifier.saveModel(modelPath + "_" + iter);
            System.out.println("done!");

            System.out.print("loading current model...");
            OnlineClassifier avgPerceptron = onlineClassifier.loadModel(modelPath + "_" + iter);
            System.out.println("done!");

            GraphBasedParser parser = new GraphBasedParser(avgPerceptron, possibleLabels);

            int labelCorrect = 0;
            int unlabelCorrect = 0;
            int allDeps = 0;

            System.out.print("\nParsing dev file with Eisner 1st order algorithm...");
            labelCorrect = 0;
            unlabelCorrect = 0;
            allDeps = 0;
            senCount = 0;
            BufferedWriter writer = new BufferedWriter(new FileWriter(outPath + "_iter" + iter));

            long start = System.currentTimeMillis();
            for (Sentence sentence : devSentences) {
                Sentence parseTree = parser.eisner1stOrder(sentence, true);
                writer.write(parseTree.toString());
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }

                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch) && !punctuations.contains(sentence.pos(ch))) {
                        allDeps++;
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);
                        int argmax = parseTree.head(ch);

                        try {
                            String bestLabel = parseTree.label(ch);

                            if (argmax == goldHead) {
                                unlabelCorrect++;
                                if (bestLabel.equals(goldLabel))
                                    labelCorrect++;
                            }
                        } catch (Exception ex) {
                            System.out.print("Why?");
                        }
                    }
                }
            }
            writer.flush();
            writer.close();
            long end = System.currentTimeMillis();
            double timeSec = (1.0 * (end - start)) / devSentences.size();
            System.out.println("");
            System.out.println("time for each sentence: " + timeSec);


            double labeledAccuracy = 100.0 * labelCorrect / allDeps;
            double unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));


            System.out.print("\nParsing dev file with Eisner 1st order algorithm...");
            labelCorrect = 0;
            unlabelCorrect = 0;
            allDeps = 0;
            senCount = 0;
            writer = new BufferedWriter(new FileWriter(outPath + "_2nd_order_iter" + iter));

            start = System.currentTimeMillis();
            for (Sentence sentence : devSentences) {
                Sentence parseTree = parser.eisner2ndOrder(sentence, true, useHandCraftedRules, false, softConstraint);
                writer.write(parseTree.toString());
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }

                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch) && !punctuations.contains(sentence.pos(ch))) {
                        allDeps++;
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);
                        int argmax = parseTree.head(ch);

                        try {
                            String bestLabel = parseTree.label(ch);

                            if (argmax == goldHead) {
                                unlabelCorrect++;
                                if (bestLabel.equals(goldLabel))
                                    labelCorrect++;
                            }
                        } catch (Exception ex) {
                            System.out.print("Why?");
                        }
                    }
                }
            }
            writer.flush();
            writer.close();
            end = System.currentTimeMillis();
            timeSec = (1.0 * (end - start)) / devSentences.size();
            System.out.println("");
            System.out.println("time for each sentence: " + timeSec);


            labeledAccuracy = 100.0 * labelCorrect / allDeps;
            unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));


            avgPerceptron = null;
        }
    }


    public static void train2ndOrder(String trainPath, ArrayList<Sentence> devSentences, ArrayList<String> possibleLabels,
                                     OnlineClassifier onlineClassifier, String modelPath, int maxIter, String outPath, boolean useHandCraftedRules,
                                     boolean trainPartial, int insertConstraintIter, double minDepProp, boolean iterativeConstraint, int resetPeriod, boolean alwaysPartial, boolean softConstraint, boolean secondOrderPartial) throws Exception {
        initializePuncs();
        ArrayList<Sentence> trainSentences = MSTReader.readSentences(trainPath, false);

        for (int iter = 0; iter < maxIter; iter++) {
            int numDep = 0;
            double correct = 0;
            GraphBasedParser trainParser = new GraphBasedParser(onlineClassifier, possibleLabels);

            System.out.println("*********************************************************");
            System.out.println("iteration: " + iter);
            int senCount = 0;

            for (Sentence sentence : trainSentences) {
                senCount++;
                if (senCount % 1000 == 0 || (secondOrderPartial && senCount%100==0)) {
                    System.out.print(senCount + "...");
                }

                if(!isProjective(sentence))
                    continue;

                boolean isCompleteTree = true;
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (!sentence.hasHead(ch)) {
                        isCompleteTree = false;
                        break;
                    }
                }
                if (isCompleteTree ) {
                    Sentence parseTree = trainParser.eisner2ndOrder(sentence, false, useHandCraftedRules, false, false);
                    boolean theSame = true;
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        numDep++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        int argmax = parseTree.head(ch);
                        if (argmax != goldHead) {
                            theSame = false;
                        } else
                            correct++;
                    }

                    if (!theSame) {
                        HashMap<String, Double> features = new HashMap<String, Double>();
                        ArrayList<Integer>[] rightDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] leftDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] goldRightDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] goldLeftDeps = new ArrayList[sentence.length()];

                        for (int ch = 0; ch < sentence.length(); ch++) {
                            rightDeps[ch] = new ArrayList<Integer>();
                            leftDeps[ch] = new ArrayList<Integer>();
                            goldRightDeps[ch] = new ArrayList<Integer>();
                            goldLeftDeps[ch] = new ArrayList<Integer>();
                        }

                        for (int ch = 1; ch < sentence.length(); ch++) {
                            int head = parseTree.head(ch);
                            if (ch > head)
                                rightDeps[head].add(ch);
                            else
                                leftDeps[head].add(ch);

                            ArrayList<String> feats = FeatureExtractor.extract1stOrderFeatures(sentence, head, ch);
                            for (String feat : feats) {
                                if (!features.containsKey(feat))
                                    features.put(feat, -1.0);
                                else
                                    features.put(feat, -1.0 + features.get(feat));
                            }

                            int goldHead = sentence.head(ch);
                            if (ch > goldHead)
                                goldRightDeps[goldHead].add(ch);
                            else
                                goldLeftDeps[goldHead].add(ch);

                            feats = FeatureExtractor.extract1stOrderFeatures(sentence, goldHead, ch);
                            for (String feat : feats) {
                                if (!features.containsKey(feat))
                                    features.put(feat, 1.0);
                                else
                                    features.put(feat, 1.0 + features.get(feat));
                            }
                        }

                        for (int i = 0; i < sentence.length(); i++) {
                            if (leftDeps[i].size() >= 1) {
                                for (int j = leftDeps[i].size() - 1; j >= 1; j--) {
                                    int r = leftDeps[i].get(j);
                                    int s = leftDeps[i].get(j - 1);
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, s);
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, -1.0);
                                        else
                                            features.put(feat, -1.0 + features.get(feat));
                                    }
                                }
                                ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, leftDeps[i].get(leftDeps[i].size() - 1));
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, -1.0);
                                    else
                                        features.put(feat, -1.0 + features.get(feat));
                                }
                            }

                            if (rightDeps[i].size() >= 1) {
                                for (int j = 0; j < rightDeps[i].size() - 1; j++) {
                                    int r = rightDeps[i].get(j);
                                    int t = rightDeps[i].get(j + 1);
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, t);
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, -1.0);
                                        else
                                            features.put(feat, -1.0 + features.get(feat));
                                    }
                                }
                                ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, rightDeps[i].get(0));
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, -1.0);
                                    else
                                        features.put(feat, -1.0 + features.get(feat));

                                }
                            }

                            if (goldLeftDeps[i].size() >= 1) {
                                for (int j = goldLeftDeps[i].size() - 1; j >= 1; j--) {
                                    int r = goldLeftDeps[i].get(j);
                                    int s = goldLeftDeps[i].get(j - 1);
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, s);
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, 1.0);
                                        else
                                            features.put(feat, 1.0 + features.get(feat));
                                    }
                                }
                                ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, goldLeftDeps[i].get(goldLeftDeps[i].size() - 1));
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, 1.0);
                                    else
                                        features.put(feat, 1.0 + features.get(feat));
                                }
                            }

                            if (goldRightDeps[i].size() >= 1) {
                                for (int j = 0; j < goldRightDeps[i].size() - 1; j++) {
                                    int r = goldRightDeps[i].get(j);
                                    int t = goldRightDeps[i].get(j + 1);
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, t);
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, 1.0);
                                        else
                                            features.put(feat, 1.0 + features.get(feat));
                                    }
                                }
                                ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, goldRightDeps[i].get(0));
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, 1.0);
                                    else
                                        features.put(feat, 1.0 + features.get(feat));
                                }
                            }
                        }

                        for (String feat : features.keySet()) {
                            double value = features.get(feat);
                            if (value != 0.0)
                                onlineClassifier.updateWeight(feat, value);
                        }
                    }
                    onlineClassifier.incrementIteration();
                } else if (secondOrderPartial && (alwaysPartial ||  iter >= insertConstraintIter)) {

                    boolean[][] stretch=new boolean[sentence.length()][sentence.length()];
                    for(int i=0;i<sentence.length();i++){
                        for(int j=i;j<sentence.length();j++){
                              if((i==j || stretch[i][j-1]) && sentence.head(j)!=-1) {
                                  stretch[i][j] = true;
                                  stretch[j][i] = true;
                              }   else{
                                  stretch[i][j] = false;
                                  stretch[j][i] = false;
                              }

                        }
                    }

                    Sentence parseTree = trainParser.eisner2ndOrder(sentence, false, useHandCraftedRules, false, false);
                    boolean theSame = true;
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        numDep++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        int argmax = parseTree.head(ch);
                        if (goldHead != -1 && argmax != goldHead) {
                            theSame = false;
                        } else
                            correct++;
                    }

                    //todo
                    if (!theSame) {
                        HashMap<String, Double> features = new HashMap<String, Double>();
                        ArrayList<Integer>[] rightDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] leftDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] goldRightDeps = new ArrayList[sentence.length()];
                        ArrayList<Integer>[] goldLeftDeps = new ArrayList[sentence.length()];

                        for (int ch = 0; ch < sentence.length(); ch++) {
                            rightDeps[ch] = new ArrayList<Integer>();
                            leftDeps[ch] = new ArrayList<Integer>();
                            goldRightDeps[ch] = new ArrayList<Integer>();
                            goldLeftDeps[ch] = new ArrayList<Integer>();
                        }

                        // extracting 1st order partial features
                        for (int ch = 1; ch < sentence.length(); ch++) {
                            int goldHead = sentence.head(ch);

                            int head = parseTree.head(ch);
                            if (ch > head)
                                rightDeps[head].add(ch);
                            else
                                leftDeps[head].add(ch);

                            //update if and only if the gold data has that partial first order dependency
                            if (goldHead != -1) {
                                ArrayList<String> feats = FeatureExtractor.extract1stOrderFeatures(sentence, head, ch);
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, -1.0);
                                    else
                                        features.put(feat, -1.0 + features.get(feat));
                                }

                                if (ch > goldHead)
                                    goldRightDeps[goldHead].add(ch);
                                else
                                    goldLeftDeps[goldHead].add(ch);

                                feats = FeatureExtractor.extract1stOrderFeatures(sentence, goldHead, ch);
                                for (String feat : feats) {
                                    if (!features.containsKey(feat))
                                        features.put(feat, 1.0);
                                    else
                                        features.put(feat, 1.0 + features.get(feat));
                                }
                            }
                        }


                        // extracting left-hand second order features
                        //todo
                        for (int i = 0; i < sentence.length(); i++) {
                            if (leftDeps[i].size() >= 1) {
                                for (int j = leftDeps[i].size() - 1; j >= 1; j--) {
                                    int r = leftDeps[i].get(j);
                                    int s = leftDeps[i].get(j - 1);

                                    // first case
                                    boolean hasStretch=stretch[r][s];

                                    boolean isNegativeExample = false;
                                    int goldRHead = sentence.head(r);
                                    int goldSHead = sentence.head(s);

                                    //second-case   and third case
                                    if ((goldRHead!=i && goldRHead!=-1) || (goldSHead!=i && goldSHead!=-1))
                                        isNegativeExample=true;


                                    if (isNegativeExample || hasStretch) {
                                        //System.out.println(i+":"+r+":"+s+":left->-1");
                                        ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, s);
                                        for (String feat : feats) {
                                            if (!features.containsKey(feat))
                                                features.put(feat, -1.0);
                                            else
                                                features.put(feat, -1.0 + features.get(feat));
                                        }
                                    }
                                }


                                // second case
                                int realHead= sentence.head(leftDeps[i].get(leftDeps[i].size()-1));
                                boolean wrongHead=false;
                                if ((realHead!=-1 && realHead!=i) || sentence.head(i)==leftDeps[i].get(leftDeps[i].size()-1))
                                    wrongHead=true;

                                // first case
                                boolean hasStretch = false;
                                if(goldLeftDeps[i].size()>0) {
                                    hasStretch = stretch[i ][goldLeftDeps[i].get(goldLeftDeps[i].size() - 1)];
                                    if (Math.abs(i - goldLeftDeps[i].get(goldLeftDeps[i].size() - 1)) == 1 && realHead!=-1)
                                        hasStretch = true;
                                }


                                if (hasStretch || wrongHead) {
                                    //System.out.println(i+":"+"0:"+ leftDeps[i].get(leftDeps[i].size() - 1)+":left->-1");
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, leftDeps[i].get(leftDeps[i].size() - 1));
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, -1.0);
                                        else
                                            features.put(feat, -1.0 + features.get(feat));
                                    }
                                }
                            }

                            //todo
                            if (goldLeftDeps[i].size() >= 1) {
                                for (int j = goldLeftDeps[i].size() - 1; j >= 1; j--) {
                                    int r = goldLeftDeps[i].get(j);
                                    int s = goldLeftDeps[i].get(j - 1);


                                    boolean hasStretch=stretch[r][s];

                                    if(hasStretch) {
                                        //System.out.println(i+":"+r+":"+s+":left->+1");
                                        ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, s);
                                        for (String feat : feats) {
                                            if (!features.containsKey(feat))
                                                features.put(feat, 1.0);
                                            else
                                                features.put(feat, 1.0 + features.get(feat));
                                        }
                                    }
                                }


                                boolean hasStretch = stretch[i][goldLeftDeps[i].get(goldLeftDeps[i].size() - 1)];
                                if(Math.abs(i-goldLeftDeps[i].get(goldLeftDeps[i].size() - 1))==1)
                                               hasStretch=true;

                                 if(hasStretch) {
                                     //System.out.println(i+":"+"0"+":"+goldLeftDeps[i].get(goldLeftDeps[i].size() - 1)+":left->+1");
                                     ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, goldLeftDeps[i].get(goldLeftDeps[i].size() - 1));
                                     for (String feat : feats) {
                                         if (!features.containsKey(feat))
                                             features.put(feat, 1.0);
                                         else
                                             features.put(feat, 1.0 + features.get(feat));
                                     }
                                 }
                            }

                            // extracting right-hand second order features
                            //todo
                            if (rightDeps[i].size() >= 1) {
                                for (int j = 0; j < rightDeps[i].size() - 1; j++) {
                                    int r = rightDeps[i].get(j);
                                    int t = rightDeps[i].get(j + 1);


                                    // first case
                                    boolean hasStretch=stretch[r][t];

                                    boolean isNegativeExample = false;
                                    int goldRHead = sentence.head(r);
                                    int goldTHead = sentence.head(t);

                                    //second-case   and third case
                                    if ((goldRHead!=i && goldRHead!=-1) || (goldTHead!=i && goldTHead!=-1))
                                        isNegativeExample=true;


                                    if(isNegativeExample || hasStretch) {
                                        //System.out.println(i+":"+r+":"+t+":right->-1");
                                        ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, t);
                                        for (String feat : feats) {
                                            if (!features.containsKey(feat))
                                                features.put(feat, -1.0);
                                            else
                                                features.put(feat, -1.0 + features.get(feat));
                                        }
                                    }
                                }

                                // second case
                                int realHead=sentence.head(rightDeps[i].get(0));
                                boolean wrongHead=false;
                                if ((realHead!=-1 && realHead!=i) || sentence.head(i)==rightDeps[i].get(0))
                                    wrongHead=true;

                                // first case
                                boolean hasStretch = false;
                                if(goldRightDeps[i].size()>0) {
                                    hasStretch = stretch[i+1][goldRightDeps[i].get(0)];
                                    if (Math.abs(i - goldRightDeps[i].get(0)) == 1 && realHead!=-1)
                                        hasStretch = true;
                                }

                                if(hasStretch || wrongHead) {
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, rightDeps[i].get(0));
                                    //System.out.println(i+":"+"0"+":"+ rightDeps[i].get(0)+":right->-1");
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, -1.0);
                                        else
                                            features.put(feat, -1.0 + features.get(feat));
                                    }
                                }
                            }

                            //todo
                            if (goldRightDeps[i].size() >= 1) {
                                for (int j = 0; j < goldRightDeps[i].size() - 1; j++) {
                                    int r = goldRightDeps[i].get(j);
                                    int t = goldRightDeps[i].get(j + 1);

                                    // first case
                                    boolean hasStretch=stretch[r][t];

                                    if(hasStretch) {
                                        //System.out.println(i+":"+r+":"+t+":right->+1");
                                        ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, r, t);
                                        for (String feat : feats) {
                                            if (!features.containsKey(feat))
                                                features.put(feat, 1.0);
                                            else
                                                features.put(feat, 1.0 + features.get(feat));
                                        }
                                    }
                                }

                                boolean hasStretch = stretch[i+1][goldRightDeps[i].get(0)];
                                if(Math.abs(i-goldRightDeps[i].get(0))==1)
                                    hasStretch=true;

                                if(hasStretch) {
                                    ArrayList<String> feats = FeatureExtractor.extract2ndOrderFeatures(sentence, i, 0, goldRightDeps[i].get(0));
                                    //System.out.println(i+":"+"0"+":"+goldRightDeps[i].get(0)+":right->+1");
                                    for (String feat : feats) {
                                        if (!features.containsKey(feat))
                                            features.put(feat, 1.0);
                                        else
                                            features.put(feat, 1.0 + features.get(feat));
                                    }
                                }
                            }
                        }
                        for (String feat : features.keySet()) {
                            double value = features.get(feat);
                            if (value != 0.0)
                                onlineClassifier.updateWeight(feat, value);
                        }
                    }
                //    if(!theSame)
                   //     System.out.println("------");
                    onlineClassifier.incrementIteration();
                } else if (alwaysPartial || (trainPartial && iter >= insertConstraintIter)) {
                    /// just train first order factors
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        if (sentence.hasHead(ch)) {
                            numDep++;
                            // finding the best head
                            int goldHead = sentence.head(ch);
                            String goldLabel = sentence.label(ch);

                            int argmax = 0;
                            String bestLabel = "";
                            double max = Double.NEGATIVE_INFINITY;

                            for (String label : possibleLabels) {
                                for (int h = 0; h < sentence.length(); h++) {
                                    double score = onlineClassifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, h, ch), false);
                                    if (score > max) {
                                        max = score;
                                        bestLabel = label;
                                        argmax = h;
                                    }
                                }
                            }

                            if (argmax != goldHead || bestLabel.equals(goldLabel)) {
                                ArrayList<String> predictedFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, argmax, ch);
                                ArrayList<String> goldFeatures = FeatureExtractor.extract1stOrderFeatures(sentence, goldHead, ch);

                                for (String predicted : predictedFeatures)
                                    onlineClassifier.updateWeight(predicted, -1);
                                for (String gold : goldFeatures)
                                    onlineClassifier.updateWeight(gold, 1);
                            } else {
                                correct++;
                            }
                        }
                    }
                    onlineClassifier.incrementIteration();
                }
            }
            System.out.println("");
            double accuracy = 100.0 * correct / numDep;
            System.out.println("size : " + onlineClassifier.size());
            System.out.println("accuracy : " + accuracy);

            System.out.print("\nsaving current model...");
            onlineClassifier.saveModel(modelPath + "_" + iter);
            System.out.println("done!");

            System.out.print("loading current model...");

            OnlineClassifier decoder = onlineClassifier.loadModel(modelPath + "_" + iter);
            System.out.println("done!");

            GraphBasedParser parser = new GraphBasedParser(decoder, possibleLabels);

            System.out.print("\nParsing dev file...");


            System.out.print("\nParsing dev file with 2nd order model...");

            int labelCorrect = 0;
            int unlabelCorrect = 0;
            int allDeps = 0;
            senCount = 0;
            BufferedWriter writer = new BufferedWriter(new FileWriter(outPath + "_iter_2nd_" + iter));

            long start = System.currentTimeMillis();
            for (Sentence sentence : devSentences) {
                Sentence parseTree = parser.eisner2ndOrder(sentence, true, true, false, false);
                writer.write(parseTree.toString());
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }

                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch) && !punctuations.contains(sentence.pos(ch))) {
                        allDeps++;
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);
                        int argmax = parseTree.head(ch);

                        try {
                            String bestLabel = parseTree.label(ch);

                            if (argmax == goldHead) {
                                unlabelCorrect++;
                                if (bestLabel.equals(goldLabel))
                                    labelCorrect++;
                            }
                        } catch (Exception ex) {
                            System.out.print("Why?");
                        }
                    }
                }
            }
            writer.flush();
            writer.close();
            long end = System.currentTimeMillis();
            double timeSec = (1.0 * (end - start)) / devSentences.size();
            System.out.println("");
            System.out.println("time for each sentence: " + timeSec);


            double labeledAccuracy = 100.0 * labelCorrect / allDeps;
            double unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));


            // todo parsing with constraints
            if (insertConstraintIter == iter + 1 || (iterativeConstraint && (insertConstraintIter - iter - 1) % resetPeriod == 0)) {

                System.out.print("\nresetting data for parsing with constraints... ");

                trainSentences = MSTReader.readSentences(trainPath, false);
                System.out.print("\nparsing with constraints... ");
                int numAll = 0;
                for (int ins = 0; ins < trainSentences.size(); ins++) {
                    if (ins % 1000 == 0) {
                        System.out.print(ins + "...");
                    }
                    Sentence sentence = trainSentences.get(ins);
                    int numDeps = 0;
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        if (sentence.hasHead(ch)) {
                            numDeps++;
                        }
                    }
                    if ((numDeps < sentence.length() - 1) && ((double) numDeps / (sentence.length() - 1)) > minDepProp) {
                        Sentence parseTree = parser.eisner2ndOrder(sentence, true, useHandCraftedRules, true, softConstraint);
                        sentence.setHeads(parseTree.getHeads());
                        sentence.setLabels(parseTree.getLabels());
                        numAll++;
                    }
                }
                System.out.println("\n added " + numAll + " trees");


                System.out.print("saving full trees... ");
                String filePath = modelPath + ".full_trees.iter" + (insertConstraintIter - iter - 1) / resetPeriod;
                BufferedWriter fullTreeWriter = new BufferedWriter(new FileWriter(filePath));
                for (Sentence sentence : trainSentences) {
                    senCount++;
                    if (senCount % 1000 == 0) {
                        System.out.print(senCount + "...");
                    }
                    boolean isCompleteTree = true;
                    for (int ch = 1; ch < sentence.length(); ch++) {
                        if (!sentence.hasHead(ch)) {
                            isCompleteTree = false;
                            break;
                        }
                    }
                    if (isCompleteTree && isProjective(sentence)) {
                        StringBuilder wOutput = new StringBuilder();
                        StringBuilder pOutput = new StringBuilder();
                        StringBuilder lOutput = new StringBuilder();
                        StringBuilder hOutput = new StringBuilder();
                        for (int i = 1; i < sentence.length(); i++) {
                            wOutput.append(sentence.word(i) + "\t");
                            pOutput.append(sentence.pos(i) + "\t");
                            lOutput.append("_" + "\t");
                            hOutput.append(sentence.head(i) + "\t");
                        }
                        fullTreeWriter.write(wOutput.toString().trim() + "\n");
                        fullTreeWriter.write(pOutput.toString().trim() + "\n");
                        fullTreeWriter.write(lOutput.toString().trim() + "\n");
                        fullTreeWriter.write(hOutput.toString().trim() + "\n\n");
                    }

                }
                fullTreeWriter.flush();
                fullTreeWriter.close();
                System.out.println("\n Saved to " + filePath);
            }
            System.out.println("");

            decoder = null;

        }
    }

    private static boolean isPunc(String pos) {
        return (pos.equals(".") || pos.equals(",") || pos.equals(":") || pos.equals("(") || pos.equals(")") || pos.equals("-LRB-") || pos.equals("-RRB-")
                || pos.equals("#") || pos.equals("$") || pos.equals("''") || pos.equals("``"));
    }

    public static boolean isProjective(Sentence sentence) {
        for (int dep1 = 1; dep1 < sentence.length(); dep1++) {
            int head1 = sentence.head(dep1);
            for (int dep2 = 1; dep2 < sentence.length(); dep2++) {
                int head2 = sentence.head(dep2);
                if (head1 == -1 || head2 == -1)
                    continue;
                if (dep1 > head1 && head1 != head2) {
                    if (dep1 > head2 && dep1 < dep2 && head1 < head2)
                        return false;
                    if (dep1 < head2 && dep1 > dep2 && head1 < dep2)
                        return false;
                }
                if (dep1 < head1 && head1 != head2) {
                    if (head1 > head2 && head1 < dep2 && dep1 < head2)
                        return false;
                    if (head1 < head2 && head1 > dep2 && dep1 < dep2)
                        return false;
                }
            }
        }
        return true;
    }

}
