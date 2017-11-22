package survey;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Survey {

    public static void main(String[] args) throws Exception {
        new DecisionTreeIris();
        new KnnIris();
    }
    
}
