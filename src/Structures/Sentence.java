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

    public Sentence(String[] words, String[] tags){
        this.words=words;
        this.tags = tags;
        heads=new int[words.length];
        labels=new String[words.length];
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
    }

    public Sentence(String[] words, String[] tags,int[] heads,String[] labels){
        this.words=words;
        this.tags = tags;
        this.heads=heads;
        this.labels=labels;
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
}
