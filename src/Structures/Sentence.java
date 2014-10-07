package Structures;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 9/16/14
 * Time: 11:36 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Sentence {
    String[] words;
    String[] tags;
    int[] heads;
    String[] labels;
    public double[] confidence;

    public Sentence(String[] words, String[] tags){
        this.words=words;
        this.tags = tags;
        heads=new int[words.length];
        labels=new String[words.length];
        confidence=new double[words.length];

        heads[0]=-1;
        labels[0]="";
        for(int i=1;i<words.length;i++){
           heads[i]=-1;
            labels[i]="";
        }
    }

    public Sentence(String[] words, String[] tags,int[] heads){
        this.words=words;
        this.tags = tags;
        this.heads=heads;
        labels=new String[words.length];
        labels[0]="";
        for(int i=1;i<words.length;i++){
            labels[i]="";
        }
        confidence=new double[words.length];

    }

    public Sentence(String[] words, String[] tags,int[] heads,String[] labels){
        this.words=words;
        this.tags = tags;
        this.heads=heads;
        this.labels=labels;
        confidence=new double[words.length];
    }

    public Sentence(String[] words, String[] tags,int[] heads,String[] labels,double[] confidence){
        this.words=words;
        this.tags = tags;
        this.heads=heads;
        this.labels=labels;
        this.confidence=confidence;
    }


    public int length(){
        return words.length;
    }

    public String word(int index){
        return words[index];
    }

    public String pos(int index){
        return tags[index];
    }

    public int head(int index){
        return heads[index];
    }

    public String label(int index){
        return labels[index];
    }

    public boolean hasHead(int index){
        return heads[index]>=0;
    }

    public String[] getWords() {
        return words;
    }

    public String[] getTags() {
        return tags;
    }

    public void setHeads(int[] heads) {
        this.heads = heads;
    }

    public void setLabels(String[] labels) {
        this.labels = labels;
    }

    public int[] getHeads() {
        return heads;
    }

    public String[] getLabels() {
        return labels;
    }

    @Override
    public String toString(){
        StringBuilder wordBuilder=new StringBuilder();
        StringBuilder tagBuilder=new StringBuilder();
        StringBuilder labelBuilder=new StringBuilder();
        StringBuilder headBuilder=new StringBuilder();

        for(int i=1;i<words.length;i++){
            wordBuilder.append(words[i]+"\t");
            tagBuilder.append(tags[i]+"\t");
            labelBuilder.append(labels[i]+"\t");
            headBuilder.append(heads[i]+"\t");
        }
           if(labelBuilder.toString().trim().length()>0)
        return wordBuilder.toString().trim()+"\n"+tagBuilder.toString().trim()+"\n"+labelBuilder.toString().trim()+"\n"+headBuilder.toString().trim()+"\n\n";
        return wordBuilder.toString().trim()+"\n"+tagBuilder.toString().trim()+"\n"+headBuilder.toString().trim()+"\n\n";
    }
}
