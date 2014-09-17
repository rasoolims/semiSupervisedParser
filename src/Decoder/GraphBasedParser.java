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
        int dimension = classifier.dimension();
        double[][][] scores = new double[l][l][labels.size()];

        int[] finalDeps = new int[l];
        finalDeps[0] = -1;
        String[] finalLabels = new String[l];
        finalLabels[0] = "";

        // getting first-order attachment scores
        for (int i = 0; i < l; i++) {
            for (int j = i+1; j < l; j++) {
                    for (int d = 0; d < labels.size(); d++) {
                        String label = labels.get(d);
                        scores[i][j][d] = classifier.score(FeatureExtractor.extractFeatures(sentence, i, j, label, dimension), decode);
                        scores[j][i][d] = classifier.score(FeatureExtractor.extractFeatures(sentence, j,i, label, dimension), decode);
                    }
            }
        }

        /**
         direction: 0=right, 1=left
         completeness: 0=incomplete, 1=complete
         **/

        int right=0;
        int left=1;

        int complete=1;
        int incomplete=0;

        double[][][][] c = new double[l][l][2][2];
        // back pointer for dependencies
        int[][][][] bd = new int[l][l][2][2];
        // back pointer for dependency labels
        String[][][][] bl = new String[l][l][2][2];
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
                    String bestRightLabel = "";
                    double bestRightScore = Double.NEGATIVE_INFINITY;
                    String bestLeftLabel = "";
                    double bestLeftScore = Double.NEGATIVE_INFINITY;

                    // finding best labels
                    for (int d = 0; d < labels.size(); d++) {
                        String label = labels.get(d);
                        double score = scores[t][s][d];
                        if (score > bestLeftScore) {
                            bestLeftScore = score;
                           bestLeftLabel  = label;
                        }

                        score = scores[s][t][d];
                        if (score > bestRightScore) {
                            bestRightScore = score;
                            bestRightLabel = label;
                        }
                    }

                    double newLeftValue = c[s][r][right][complete] + c[r + 1][t][left][complete] + bestLeftScore;
                    if (newLeftValue > c[s][t][left][incomplete]) {
                        c[s][t][left][incomplete] = newLeftValue;
                        bd[s][t][left][incomplete] = r;
                        bl[s][t][left][incomplete] = bestLeftLabel;
                    }

                    double newRightValue = c[s][r][right][complete] + c[r + 1][t][left][complete] + bestRightScore;
                    if (newRightValue > c[s][t][right][incomplete]) {
                        c[s][t][right][incomplete] = newRightValue;
                        bd[s][t][right][incomplete] = r;
                        bl[s][t][right][incomplete] = bestRightLabel;
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

        retrieveDeps(bd, bl, 0, l - 1, 0, 1, finalLabels, finalDeps);

       return new Sentence(sentence.getWords(),sentence.getTags(),finalDeps,finalLabels);
    }

    public void retrieveDeps(int[][][][] bd, String[][][][] bl, int s, int t, int direction,
                             int completeness, String[] finalLabels, int[] finalDeps) {
        if (s == t)
            return;
        int r = bd[s][t][direction][completeness];
        if (completeness == 1) {
            if (direction == 0) {
                retrieveDeps(bd, bl, s, r, 0, 0, finalLabels, finalDeps);
                retrieveDeps(bd, bl, r, t, 0, 1, finalLabels, finalDeps);
            } else {
                retrieveDeps(bd, bl, s, r, 1, 1, finalLabels, finalDeps);
                retrieveDeps(bd, bl, r, t, 1, 0, finalLabels, finalDeps);
            }
        } else {
            if (direction == 0) {
                finalDeps[t] = s;
                retrieveDeps(bd, bl, s, r, 0, 1, finalLabels, finalDeps);
                retrieveDeps(bd, bl, r + 1, t, 1, 1, finalLabels, finalDeps);
                finalLabels[t] = bl[s][t][direction][completeness];

            } else {
                finalDeps[s] = t;
                retrieveDeps(bd, bl, s, r, 0, 1, finalLabels, finalDeps);
                retrieveDeps(bd, bl, r + 1, t, 1, 1, finalLabels, finalDeps);
                finalLabels[s] = bl[s][t][direction][completeness];
            }
        }
    }
}
