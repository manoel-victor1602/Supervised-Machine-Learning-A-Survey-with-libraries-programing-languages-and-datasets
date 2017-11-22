package survey;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class KnnIris {
    
    KnnIris() throws Exception{
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("src/survey/iris_data.arff");
        Instances instances = dataSource.getDataSet();
        
        instances.setClassIndex(4);
        
        IBk KNN = new IBk();
        KNN.buildClassifier(instances);
        
        
        Evaluation evaluation = new Evaluation(instances);
        
        evaluation.crossValidateModel(KNN, instances, 10, new Debug.Random());
        
        System.out.println("KNN CrossValidation: " + evaluation.pctCorrect());
    }
    
}
