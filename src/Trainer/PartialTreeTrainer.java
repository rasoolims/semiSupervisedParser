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
                             AveragedPerceptron perceptron, String modelPath, int maxIter, boolean trainStructuredForFullTrees,String outPath) throws Exception {

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

        int dimension = perceptron.dimension();

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
                boolean isCompleteTree=true;
                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (!sentence.hasHead(ch)) {
                        isCompleteTree=false;
                        break;
                    }
                }
                if(!isCompleteTree || !trainStructuredForFullTrees){
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
                                    double score = perceptron.score(FeatureExtractor.extractFeatures(sentence, h, ch, label, dimension), false);
                                    if (score > max) {
                                        max = score;
                                        bestLabel = label;
                                        argmax = h;
                                    }
                                }
                            }

                            if (argmax != goldHead || bestLabel.equals(goldLabel)) {
                                Object[] predictedFeatures = FeatureExtractor.extractFeatures(sentence, argmax, ch, bestLabel, dimension);
                                Object[] goldFeatures = FeatureExtractor.extractFeatures(sentence, goldHead, ch,bestLabel.equals("")?"": goldLabel, dimension);

                                for (int i = 0; i < predictedFeatures.length; i++) {
                                    if (predictedFeatures[i] instanceof String) {
                                        if (!predictedFeatures[i].equals(goldFeatures[i])) {
                                            perceptron.updateWeight(i, (String) predictedFeatures[i],sentence.confidence[ch]*-1.0);
                                            perceptron.updateWeight(i, (String) goldFeatures[i],sentence.confidence[ch]* 1.0);
                                        }
                                    } else {
                                        HashMap<String, Integer> prd = (HashMap<String, Integer>) predictedFeatures[i];
                                        HashMap<String, Integer> gold = (HashMap<String, Integer>) goldFeatures[i];

                                        for (String feat : prd.keySet()) {
                                            perceptron.updateWeight(i, feat,sentence.confidence[ch]* -prd.get(feat));
                                        }
                                        for (String feat : gold.keySet()) {
                                            perceptron.updateWeight(i, feat,sentence.confidence[ch]* gold.get(feat));
                                        }
                                    }
                                }
                            } else {
                                correct++;
                            }

                        }
                    }
                    perceptron.incrementIteration();
                } else{
                    Sentence parseTree = trainParser.eisner1stOrder(sentence, false);

                    for (int ch = 1; ch < sentence.length(); ch++) {
                        numDep++;
                        // finding the best head
                        int goldHead = sentence.head(ch);
                        String goldLabel = sentence.label(ch);
                        if(possibleLabels.size()<=1)
                            goldLabel="";

                        int argmax = parseTree.head(ch);
                        String bestLabel = parseTree.label(ch);

                        if (argmax != goldHead || (possibleLabels.size()>1 && !bestLabel.equals(goldLabel))) {
                            Object[] predictedFeatures = FeatureExtractor.extractFeatures(sentence, argmax, ch, bestLabel, dimension);
                            Object[] goldFeatures = FeatureExtractor.extractFeatures(sentence, goldHead, ch, goldLabel, dimension);

                            for (int i = 0; i < predictedFeatures.length; i++) {
                                if (predictedFeatures[i] instanceof String) {
                                    if (!predictedFeatures[i].equals(goldFeatures[i])) {
                                        perceptron.updateWeight(i, (String) predictedFeatures[i], sentence.confidence[ch]*-1.0);
                                        perceptron.updateWeight(i, (String) goldFeatures[i], sentence.confidence[ch]*1.0);
                                    }
                                } else {
                                    HashMap<String, Integer> prd = (HashMap<String, Integer>) predictedFeatures[i];
                                    HashMap<String, Integer> gold = (HashMap<String, Integer>) goldFeatures[i];

                                    for (String feat : prd.keySet()) {
                                        perceptron.updateWeight(i, feat,sentence.confidence[ch]* -prd.get(feat));
                                    }
                                    for (String feat : gold.keySet()) {
                                        perceptron.updateWeight(i, feat, sentence.confidence[ch]*gold.get(feat));
                                    }
                                }
                            }
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
                                double score = avgPerceptron.score(FeatureExtractor.extractFeatures(sentence, h, ch, label, dimension), true);
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


            labelCorrect = 0;
            unlabelCorrect = 0;
            allDeps = 0;
            senCount = 0;
            BufferedWriter writer=new BufferedWriter(new FileWriter(outPath+"_iter"+iter));

            long start=System.currentTimeMillis();
            for (Sentence sentence : devSentences) {
                Sentence parseTree = parser.eisner1stOrder(sentence, true);
                writer.write(parseTree.toString());
                senCount++;
                if (senCount % 100 == 0) {
                    System.out.print(senCount + "...");
                }

                for (int ch = 1; ch < sentence.length(); ch++) {
                    if (sentence.hasHead(ch) &&  !punctuations.contains(sentence.pos(ch))) {
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
            avgPerceptron=null;
            writer.flush();
            writer.close();
            long end=System.currentTimeMillis();
            double timeSec=(1.0*(end-start))/devSentences.size();
            System.out.println("");
            System.out.println("time for each sentence: " + timeSec);


            labeledAccuracy = 100.0 * labelCorrect / allDeps;
            unlabeledAccuracy = 100.0 * unlabelCorrect / allDeps;
            System.out.println(String.format("unlabeled: %s labeled: %s", unlabeledAccuracy, labeledAccuracy));

        }
    }

    private static boolean isPunc(String pos){
        return (pos.equals(".") || pos.equals(",") || pos.equals(":") || pos.equals("(") || pos.equals(")")  || pos.equals("-LRB-") || pos.equals("-RRB-")
                || pos.equals("#") || pos.equals("$")  || pos.equals("''")|| pos.equals("``")) ;
    }
}
