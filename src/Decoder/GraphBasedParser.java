package Decoder;

import Classifier.AveragedPerceptron;
import Structures.Sentence;

import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/17/14
 * Time: 11:06 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class GraphBasedParser {
    AveragedPerceptron classifier;
    ArrayList<String> labels;

    public GraphBasedParser(AveragedPerceptron perceptron,
                            ArrayList<String> labels) {
        this.classifier = perceptron;
        this.labels = labels;
    }

    public Sentence eisner1stOrder(Sentence sentence, boolean decode) {
        int l = sentence.length();
        double[][] scores = new double[l][l];

        int[] finalDeps = new int[l];
        finalDeps[0] = -1;

        // getting first-order attachment scores
        for (int i = 0; i < l; i++) {
            for (int j = i + 1; j < l; j++) {
                scores[i][j] =classifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, i, j), decode);
                scores[j][i] =  classifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, j, i), decode);
            }
        }

        /**
         direction: 0=right, 1=left
         completeness: 0=incomplete, 1=complete
         **/

        int right = 0;
        int left = 1;

        int complete = 1;
        int incomplete = 0;

        double[][][][] c = new double[l][l][2][2];
        // back pointer for dependencies
        int[][][][] bd = new int[l][l][2][2];
        // back pointer for dependency labels
        for (int s = 0; s < l; s++) {
            c[s][s][right][complete] = 0.0;
            c[s][s][left][complete] = 0.0;
        }

        for (int k = 1; k < l; k++) {
            for (int s = 0; s < l; s++) {
                int t = s + k;
                if (t >= l) break;


                // create incomplete items
                c[s][t][left][incomplete] = Double.NEGATIVE_INFINITY;
                c[s][t][right][incomplete] = Double.NEGATIVE_INFINITY;
                for (int r = s; r < t; r++) {
                    double bestRightScore = scores[s][t];
                    double bestLeftScore = scores[t][s];


                    double newLeftValue = c[s][r][right][complete] + c[r + 1][t][left][complete] + bestLeftScore;
                    if (newLeftValue > c[s][t][left][incomplete]) {
                        c[s][t][left][incomplete] = newLeftValue;
                        bd[s][t][left][incomplete] = r;
                    }

                    double newRightValue = c[s][r][right][complete] + c[r + 1][t][left][complete] + bestRightScore;
                    if (newRightValue > c[s][t][right][incomplete]) {
                        c[s][t][right][incomplete] = newRightValue;
                        bd[s][t][right][incomplete] = r;
                    }
                }

                // create complete spans
                c[s][t][left][complete] = Double.NEGATIVE_INFINITY;
                c[s][t][right][complete] = Double.NEGATIVE_INFINITY;
                for (int r = s; r <= t; r++) {
                    if (r < t) {
                        double newLeftScore = c[s][r][left][complete] + c[r][t][left][incomplete];
                        if (newLeftScore > c[s][t][left][complete]) {
                            c[s][t][left][complete] = newLeftScore;
                            bd[s][t][left][complete] = r;
                            //    bl[s][t][left][complete] = bl[r][t][left][0];
                        }
                    }

                    if (r > s) {
                        double newRightScore = c[s][r][right][incomplete] + c[r][t][right][complete];
                        if (newRightScore > c[s][t][right][complete]) {
                            c[s][t][right][complete] = newRightScore;
                            bd[s][t][right][complete] = r;
                            //    bl[s][t][right][complete] =bl[s][r][right][0];
                        }
                    }
                }
            }
        }

        retrieveDeps(bd,  0, l - 1, 0, 1, finalDeps);

        return new Sentence(sentence.getWords(), sentence.getTags(), finalDeps);
    }

    public Sentence eisner2ndOrder(Sentence sentence, boolean decode) {
        int l = sentence.length();

        int[] finalDeps = new int[l];
        finalDeps[0] = -1;

        /**
         direction: 0=right, 1=left
         completeness: 0=incomplete, 1=complete
         **/
        int right = 0;
        int left = 1;
        int neutral = 2;

        int complete = 1;
        int incomplete = 0;
        int rectangular = 2;

        double[][][][] c = new double[l][l][3][3];
        // back pointer for dependencies
        int[][][][] bd = new int[l][l][3][3];
        // back pointer for dependency labels

        // initialization
        for (int s = 0; s < l; s++) {
            for (int d = 0; d < 3; d++) {
                for (int cn = 0; cn < 3; cn++) {
                    c[s][s][d][cn] = 0.0;
                }
            }
        }

        double[][] scores = new double[l][l];
        // getting first-order attachment scores
        for (int i = 0; i < l; i++) {
            for (int j = i + 1; j < l; j++) {
                scores[i][j] = classifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, i, j), decode);
                scores[j][i] =classifier.score(FeatureExtractor.extract1stOrderFeatures(sentence, j, i), decode);
            }
        }

        for (int k = 1; k < l; k++) {
            for (int s = 0; s < l; s++) {
                int t = s + k;
                if (t >= l) break;

                // creating sibling items
                double maxValue=Double.NEGATIVE_INFINITY;
                int maxR=-1;
                for (int r = s; r < t; r++) {
                    double newValue = c[s][r][right][complete] + c[r + 1][t][left][complete];
                    if (newValue > maxValue) {
                        maxValue = newValue;
                        maxR = r;
                    }
                }
                if(maxValue==Double.NEGATIVE_INFINITY){
                    System.out.print("here");
                }
                c[s][t][neutral][rectangular] = maxValue;
                bd[s][t][neutral][rectangular]=maxR;

                c[s][t][left][incomplete] = c[s][t - 1][right][complete] + c[t][t][left][complete] + classifier.score(FeatureExtractor.extract2ndOrderFeatures(sentence, t, 0, s), decode)+scores[t][s];
                bd[s][t][left][incomplete] = t;
                c[s][t][right][incomplete] = c[s][s][right][complete] + c[s + 1][t][left][complete]  + classifier.score(FeatureExtractor.extract2ndOrderFeatures(sentence, s, 0, t), decode)+scores[s][t];
                bd[s][t][right][incomplete] = s;


                // second case: head picks up a pair of modifiers (through a sibling item)
                for (int r = s; r < t; r++) {
                    // first case: head picks up first modifier
                    if(r<t) {
                        double newValue = c[s][r][neutral][rectangular] + c[r][t][left][incomplete] + classifier.score(FeatureExtractor.extract2ndOrderFeatures(sentence, t, r, s), decode) + scores[t][s] ;
                        if (newValue > c[s][t][left][incomplete]) {
                            c[s][t][left][incomplete] = newValue;
                            bd[s][t][left][incomplete] = r;
                        }
                    }

                    if(r>s) {
                      double  newValue = c[s][r][right][incomplete] + c[r][t][neutral][rectangular] + classifier.score(FeatureExtractor.extract2ndOrderFeatures(sentence, s, r, t), decode) + scores[s][t] ;
                        if (newValue > c[s][t][right][incomplete]) {
                            c[s][t][right][incomplete] = newValue;
                            bd[s][t][right][incomplete] = r;
                        }
                    }
                }

                // create complete items
                double maxLeftValue=Double.NEGATIVE_INFINITY;
                int maxLeftR=s;
                int maxRightR=t;
                double maxRightValue=  Double.NEGATIVE_INFINITY;
                for (int r = s; r <= t; r++) {
                    if (r < t) {
                        double newLeftScore = c[s][r][left][complete] + c[r][t][left][incomplete];
                        if (newLeftScore >maxLeftValue) {
                            maxLeftValue = newLeftScore;
                            maxLeftR = r;
                            //    bl[s][t][left][complete] = bl[r][t][left][0];
                        }
                    }

                    if (r > s) {
                        double newRightScore = c[s][r][right][incomplete] + c[r][t][right][complete];
                        if (newRightScore > maxRightValue) {
                            maxRightValue = newRightScore;
                            maxRightR = r;
                            //    bl[s][t][right][complete] =bl[s][r][right][0];
                        }
                    }
                }
                if(maxRightValue==Double.NEGATIVE_INFINITY){
                    System.out.print("HERE!");
                }

                c[s][t][left][complete]=maxLeftValue;
                bd[s][t][left][complete]=maxLeftR;
                c[s][t][right][complete]=maxRightValue;
                bd[s][t][right][complete]=maxRightR;
            }
        }
        retrieve2ndDeps(bd, 0, l - 1, 0, 1, finalDeps);

        return new Sentence(sentence.getWords(), sentence.getTags(), finalDeps);
    }


    public void retrieveDeps(int[][][][] bd, int s, int t, int direction,
                             int completeness, int[] finalDeps) {
        if (s == t)
            return;
        int r = bd[s][t][direction][completeness];
        if (completeness == 1) {
            if (direction == 0) {
                retrieveDeps(bd, s, r, 0, 0, finalDeps);
                retrieveDeps(bd,  r, t, 0, 1, finalDeps);
            } else {
                retrieveDeps(bd,  s, r, 1, 1, finalDeps);
                retrieveDeps(bd,  r, t, 1, 0, finalDeps);
            }
        } else {
            if (direction == 0) {
                finalDeps[t] = s;
                retrieveDeps(bd,s, r, 0, 1, finalDeps);
                retrieveDeps(bd,  r + 1, t, 1, 1, finalDeps);

            } else {
                finalDeps[s] = t;
                retrieveDeps(bd, s, r, 0, 1, finalDeps);
                retrieveDeps(bd,  r + 1, t, 1, 1, finalDeps);
            }
        }
    }

    public void retrieve2ndDeps(int[][][][] bd,int s, int t, int direction,
                                int completeness, int[] finalDeps) {
        if (s == t)
            return;

        int r = bd[s][t][direction][completeness];

        if (completeness == 1) {
            if (direction == 0) {
                retrieve2ndDeps(bd, s, r, 0, 0, finalDeps);
                retrieve2ndDeps(bd, r, t, 0, 1,  finalDeps);
            } else if (direction == 1) {
                retrieve2ndDeps(bd, s, r, 1, 1, finalDeps);
                retrieve2ndDeps(bd, r, t, 1, 0,  finalDeps);
            }
        } else if (completeness == 0) {
            if (direction == 0) {
                finalDeps[t] = s;

                if (r > s && t>r ) {
                    retrieve2ndDeps(bd, s, r, 0, 0,  finalDeps);
                    retrieve2ndDeps(bd, r, t, 2, 2, finalDeps);
                } else {
                    retrieve2ndDeps(bd, s + 1, t, 1, 1,  finalDeps);
                }
            } else if (direction == 1) {
                finalDeps[s] = t;
                if (r > s && t > r ) {
                    retrieve2ndDeps(bd, s, r, 2, 2,finalDeps);
                    retrieve2ndDeps(bd, r, t, 1, 0,  finalDeps);
                } else {
                    retrieve2ndDeps(bd, s, t -1, 0, 1, finalDeps);
                }
            }
        } else {
            retrieve2ndDeps(bd, s, r, 0, 1,  finalDeps);
            retrieve2ndDeps(bd, r + 1, t, 1, 1, finalDeps);
        }
    }

    public boolean isPP(String pos) {
        return (pos.equals("TO") || pos.equals("IN") || pos.equals("ADP"));
    }

    public boolean isNOUN(String pos) {
        return (pos.startsWith("NN") || pos.equals("NOUN") || pos.equals("N"));
    }

    public boolean isVerb(String pos) {
        return (pos.startsWith("V"));
    }
}
