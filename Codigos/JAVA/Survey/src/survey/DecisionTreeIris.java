package survey;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class DecisionTreeIris {
    
    DecisionTreeIris() throws Exception{
        DataSource dataSource = new DataSource("src/survey/iris_data.arff");
        Instances instances = dataSource.getDataSet();
        
        instances.setClassIndex(4);
        
        J48 tree = new J48();
        tree.buildClassifier(instances);
        
        
        Evaluation evaluation = new Evaluation(instances);
        
        evaluation.crossValidateModel(tree, instances, 10, new Debug.Random());
        
        System.out.println("DecisionTree CrossValidation: " + evaluation.pctCorrect());
    }
    
    
}
