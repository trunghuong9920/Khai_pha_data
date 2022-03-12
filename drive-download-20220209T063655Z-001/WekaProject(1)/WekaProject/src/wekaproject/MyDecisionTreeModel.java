/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaproject;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Debug.Random;

/**
 *
 * @author Khoa
 */
public class MyDecisionTreeModel extends MyKnowledgeModel{
    J48 tree;

    public MyDecisionTreeModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    public void buildDecisionTree() throws Exception{
        //tạo tập dl huấn luyện 
       this.trainset = divideTrainTest(this.dataset, 80, false);
       this.testset = divideTrainTest(this.dataset, 80, true);
       this.trainset.setClassIndex(this.trainset.numAttributes()-1);
       this.testset.setClassIndex(this.testset.numAttributes()-1);
        //thiết lập thông số
        
        tree = new J48();
        tree.setOptions(this.model_options);
        tree.buildClassifier(this.trainset);
    }

    public void evaluateDecisionTree () throws Exception
    { 
        Random rnd = new Debug.Random (1);
        int folds = 10;
        Evaluation eval = new Evaluation (this.trainset);
        eval.crossValidateModel (tree, this.testset, folds, rnd);
        System.out.println (eval.toSummaryString( "\nKet qua danh gia mo hinh 10-fold cross-validation\n-----\n", false));
    }   
    @Override
    public String toString() {
        return tree.toSummaryString(); //To change body of generated methods, choose Tools | Templates.
    }
    
}
