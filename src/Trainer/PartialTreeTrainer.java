package Trainer;

import Classifier.AveragedPerceptron;
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
    public static void train(ArrayList<Sentence> trainSentences, ArrayList<Sentence> devSentences, ArrayList<String> possibleLabels,
                             AveragedPerceptron perceptron, String modelPath, int maxIter, boolean trainStructuredForFullTrees, String outPath) throws Exception {

        HashSet<String> punctuations = new HashSet<String>();
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

        for (int iter = 0; iter < maxIter; iter++) {
            int numDep = 0;
            double correct = 0;
            GraphBasedParser trainParser = new GraphBasedParser(perceptron, possibleLabels);

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
                                    double score = perceptron.score(FeatureExtractor.extract1stOrderFeatures(sentence, h, ch), false);
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

                                for(String predicted:predictedFeatures)
                                    perceptron.updateWeight(0,predicted,-1);
                                for(String gold:goldFeatures)
                                    perceptron.updateWeight(0,gold,1);
                            } else {
                                correct++;
                            }
                        }
                    }
                    perceptron.incrementIteration();
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

                            for(String predicted:predictedFeatures)
                                perceptron.updateWeight(0,predicted,-1);
                            for(String gold:goldFeatures)
                                perceptron.updateWeight(0,gold,1);
                        } else {
                            correct++;
                        }
                    }
                    perceptron.incrementIteration();

                }
            }
            System.out.println("");
            double accuracy = 100.0 * correct / numDep;
            System.out.println("size : " + perceptron.size());
            System.out.println("accuracy : " + accuracy);

            System.out.print("\nsaving current model...");
            perceptron.saveModel(modelPath + "_" + iter);
            System.out.println("done!");

            System.out.print("loading current model...");
            AveragedPerceptron avgPerceptron = AveragedPerceptron.loadModel(modelPath + "_" + iter);
            System.out.println("done!");

            GraphBasedParser parser = new GraphBasedParser(avgPerceptron, possibleLabels);

            System.out.print("\nParsing dev file...");

            int labelCorrect = 0;
            int unlabelCorrect = 0;
            int allDeps = 0;
            senCount = 0;

            for (Sentence sentence : devSentences) {
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch) && !punctuations.contains(sentence.pos(ch))) {
                        allDeps++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);

                        int argmax = 0;
                        String bestLabel = "";
                        double max = Double.NEGATIVE_INFINITY;

                        for (String label : possibleLabels) {
                            for (int h = 0; h < sentence.length(); h++) {
                                double score = avgPerceptron.score(FeatureExtractor.extract1stOrderFeatures(sentence, h, ch), true);
                                if (score > max) {
                                    max = score;
                                    bestLabel = label;
                                    argmax = h;
                                }
                            }
                        }

                        if (argmax == goldHead) {
                            unlabelCorrect++;
                            if (bestLabel.equals(goldLabel))
                                labelCorrect++;
                        }
                    }
                }
            }
            System.out.println("");

            double labeledAccuracy = 100.0 * labelCorrect / allDeps;
            double unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));

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


            labeledAccuracy = 100.0 * labelCorrect / allDeps;
            unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));
            avgPerceptron = null;
        }
    }


    public static void train2ndOrder(ArrayList<Sentence> trainSentences, ArrayList<Sentence> devSentences, ArrayList<String> possibleLabels,
                                     AveragedPerceptron perceptron, String modelPath, int maxIter, String outPath) throws Exception {

        HashSet<String> punctuations = new HashSet<String>();
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

        for (int iter = 0; iter < maxIter; iter++) {
            int numDep = 0;
            double correct = 0;
            GraphBasedParser trainParser = new GraphBasedParser(perceptron, possibleLabels);

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
                if (isCompleteTree) {
                    Sentence parseTree = trainParser.eisner2ndOrder(sentence, false);
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

                        for (int i = 1; i < sentence.length(); i++) {
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
                                perceptron.updateWeight(0, feat, value);
                        }
                    }
                    perceptron.incrementIteration();
                }
            }
            System.out.println("");
            double accuracy = 100.0 * correct / numDep;
            System.out.println("size : " + perceptron.size());
            System.out.println("accuracy : " + accuracy);

            System.out.print("\nsaving current model...");
            perceptron.saveModel(modelPath + "_" + iter);
            System.out.println("done!");

            System.out.print("loading current model...");
            AveragedPerceptron avgPerceptron = AveragedPerceptron.loadModel(modelPath + "_" + iter);
            System.out.println("done!");

            GraphBasedParser parser = new GraphBasedParser(avgPerceptron, possibleLabels);

            System.out.print("\nParsing dev file...");


            System.out.print("\nParsing dev file with 2nd order model...");

            int labelCorrect = 0;
            int unlabelCorrect = 0;
            int allDeps = 0;
            senCount = 0;
            BufferedWriter writer = new BufferedWriter(new FileWriter(outPath + "_iter_2nd_" + iter));

            long start = System.currentTimeMillis();
            for (Sentence sentence : devSentences) {
                //  if(senCount==78){
                //      System.out.print("HERE");
                //  }
                Sentence parseTree = parser.eisner2ndOrder(sentence, true);
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


            avgPerceptron = null;

        }
    }

    private static boolean isPunc(String pos) {
        return (pos.equals(".") || pos.equals(",") || pos.equals(":") || pos.equals("(") || pos.equals(")") || pos.equals("-LRB-") || pos.equals("-RRB-")
                || pos.equals("#") || pos.equals("$") || pos.equals("''") || pos.equals("``"));
    }
}
