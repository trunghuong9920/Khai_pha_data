/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaproject;

/**
 *
 * @author Khoa
 */
public class WekaProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
       MyDecisionTreeModel model = new MyDecisionTreeModel("D:\\data\\bank-edited.csv",null,null);
       model.buildDecisionTree();
       model.evaluateDecisionTree();
        System.out.println(model);
    }

    
    
}
