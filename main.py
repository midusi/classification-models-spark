import os
import pandas as pd
import numpy as np
import time
import logging
from typing import Tuple, Iterable, Dict
from StratifiedCrossValidator import StratifiedCrossValidator, CrossValidationResult, K_TIMING_KEY
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import StructField, StructType, FloatType
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.tuning import ParamGridBuilder
from sklearn.preprocessing import label_binarize


# Number of features to be used
NUMBER_OF_FEATURES = [10, 50, 100, 500, 1000, 5000, 10000]

# Number of folds to be used in CrossValidation
K = 5

# Name of the label column
NEW_CLASS_NAME = 'class'

# Disables lot of logging from Spark
logging.getLogger().setLevel(logging.INFO)

# Number of cores used for Cross Validation. -1 = use all
N_JOBS = -1

# Number of iterations to get statistical significance
NUMBER_OF_ITERATIONS = 2

# Spark settings
# Enable Arrow-based columnar data transfers
conf = SparkConf() \
    .setMaster("spark://master-node:7077") \
    .setAppName("Classification")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sql = SparkSession(sc)


def parse_row_for_dataframe(idx_and_row: Iterable) -> Tuple[float, DenseVector]:
    """
    Generates the necessary information to generate the Spark DataFrame.
    :param idx_and_row: A Pandas tuple containing class and features
    :return: A tuple with the class and a DenseVector with the features
    """
    (_, row) = idx_and_row
    y = row.loc[NEW_CLASS_NAME].item()
    x: pd.Series = row.drop(NEW_CLASS_NAME)
    return y, Vectors.dense(x.to_list())


def get_data_ready(x_subset_with_y: pd.DataFrame) -> SparkDataFrame:
    """
    Obtains the parsed data. Serves to avoid processing the same data several times
    :param x_subset_with_y: Features subset to use
    :return: Spark SQL DataFrame ready to use by classification models
    """
    parsed_data_for_dataframe = list(map(parse_row_for_dataframe, x_subset_with_y.iterrows()))

    data_dataframe = sql.createDataFrame(parsed_data_for_dataframe, schema=StructType([
        StructField("label", FloatType()),
        StructField("features", VectorUDT())
    ]))

    return data_dataframe


def binarize_y(y: pd.Series) -> Tuple[np.ndarray, int]:
    """
    Generates a binary array indicating the class
    :param y: Array with classes/labels
    :return: Categorical array binarized
    """
    classes = y.unique()
    return label_binarize(y, classes=classes).ravel(), classes.shape[0]


def get_x_and_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Gets X and Y from a DataFrame
    :param df: DataFrame to split
    :return: X and Y data
    """
    y, number_classes = binarize_y(df[NEW_CLASS_NAME])
    return df.loc[:, df.columns != NEW_CLASS_NAME], y, number_classes


def print_metrics(cross_result: CrossValidationResult):
    """
    Prints all the metrics (for DEBUG purposes)
    :param cross_result: Cross validation result dict
    :return A tuple with all the K timings (T3) and the mean of T3 (T4)
    """
    for estimator_name in cross_result:
        logging.info(f'Estimator {estimator_name}')
        for metric_name in cross_result[estimator_name]:
            model_or_times = cross_result[estimator_name][metric_name]
            if metric_name == K_TIMING_KEY:
                # It's a list
                logging.info(f'\t{K} folds times -> {model_or_times}')
                logging.info(f'\tT4 -> {np.mean(model_or_times)}')
            else:
                # It's a metric
                model, best_index = model_or_times
                logging.info(f'\t{metric_name}.avgMetrics -> {model.avgMetrics} | best {model.avgMetrics[best_index]}')


def run_models(data_dataframe: SparkDataFrame, number_of_features: int, number_classes: int) -> CrossValidationResult:
    """
    Run all the models in the Stratified Cross Validator
    :param data_dataframe: Data to train/test
    :param number_of_features: Number of features to use
    :param number_classes: Number of classes (needed for the Neural Network model)
    :return: Stratified Cross Validator result
    """
    # For Neural Networks
    layers = [number_of_features, 5, 4, number_classes]

    # NOTE: we get weighted as recommended in https://www.quora.com/How-do-I-compute-for-the-overall-Precision-and-\
    # Recall-Do-I-get-the-Precision-and-Recall-for-each-class-and-get-the-average
    cv = StratifiedCrossValidator(estimator=[
        NaiveBayes(),
        RandomForestClassifier(numTrees=20),
        LinearSVC(),
        MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    ],
        # IMPORTANT: do not put different parameters in ParamGrid because it will bias the training time as the .fit()
        # method uses all possible models.
        estimatorParamMaps=ParamGridBuilder().build(),
        evaluator=[
            BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol='label', rawPredictionCol='rawPrediction'),
            MulticlassClassificationEvaluator(metricName='falsePositiveRateByLabel'),  # Specificity = 1 - FPR
            MulticlassClassificationEvaluator(metricName='weightedPrecision'),
            # Sensitivity = Recall in binary classification
            MulticlassClassificationEvaluator(metricName='weightedRecall'),
            MulticlassClassificationEvaluator(metricName='accuracy'),
            MulticlassClassificationEvaluator(metricName='f1')
        ],
        numFolds=K,
        parallelism=N_JOBS
    )
    cross_result: CrossValidationResult = cv.fit(data_dataframe)

    # DEBUG: prints metrics
    # print_metrics(cross_result)

    return cross_result


def main():
    # Dataset
    # df = pd.read_csv('./Datasets/Ovarian.csv')  # TODO: use this line to get all dataset
    df = pd.read_csv('./Datasets/Ovarian.csv', nrows=25)

    # Renames label column
    df = df.rename(columns={'Class': NEW_CLASS_NAME})
    x, y, number_classes = get_x_and_y(df)

    # DEBUG
    # print('Shape', df.shape)
    # print('NaNs?', df.isnull().values.any())
    # print('x.shape', x.shape)
    # print('y.shape', y.shape)
    # print('number_classes', number_classes)

    # Final CSV file where results will be stored
    now = time.strftime('%Y-%m-%d_%H:%M:%S')
    dir_name = os.path.dirname(__file__)
    res_csv_file_path = os.path.join(dir_name, f'Results/result_{now}.csv')

    logging.info(f'The results file will be saved in {res_csv_file_path}')

    # List of results to save as CSV
    T4_KEY = 'T4'
    result_csv = pd.DataFrame([])

    for n_features in NUMBER_OF_FEATURES:
        # Gets N features
        start = time.time()
        # x_subset = x.sample(n=n_features, axis='columns', random_state=2)  # Uses seed...
        x_subset = x.sample(n=n_features, axis='columns')
        time_t1 = time.time() - start
        logging.info(f'Time to obtain {n_features} features (T1) -> {time_t1} seconds')

        # Gets features list to save in CSV
        features_names = ', '.join(x_subset.columns.values.tolist())

        # Adds class column to iterate over rows generating data
        x_subset[NEW_CLASS_NAME] = y

        # Gets DataFrame (for model like RandomForest) and a list of LabeledPoint (for SVM)
        start = time.time()
        data_dataframe = get_data_ready(x_subset)
        time_t2 = time.time() - start
        logging.info(f'Spark DataFrame creation (T2) -> {time_t2} seconds')

        # DEBUG:
        # data_dataframe.printSchema()
        # print(data_dataframe.show(10))

        # Runs all models and stores in a dict all the important data
        # n_runs = Estimator name -> {n_accuracy: [], n_f1: [], ..., n_t4: [], T5: float}
        n_runs: Dict[str, Dict] = {}
        for i in range(NUMBER_OF_ITERATIONS):
            cross_result = run_models(data_dataframe, n_features, number_classes)

            for estimator_name in cross_result:
                # If needed, adds estimator to dict
                if estimator_name not in n_runs:
                    n_runs[estimator_name] = {}

                for field_name in cross_result[estimator_name]:
                    model_with_metrics_or_times = cross_result[estimator_name][field_name]

                    # Check if it's a timing or a metric
                    if field_name == K_TIMING_KEY:
                        # It's a list. Adds mean of T4 to list
                        # DEBUG:
                        # logging.info(f'K timing -> {model_with_metrics_or_times}')
                        field_name = T4_KEY
                        value = np.mean(model_with_metrics_or_times)
                    else:
                        # It's a metric. Adds metric to list
                        cross_model, best_model_idx = model_with_metrics_or_times
                        value = cross_model.avgMetrics[best_model_idx]

                        print(f'avgMetrics of {field_name} for {estimator_name} -> {value} (best from '
                              f'{cross_model.avgMetrics})')

                        # Specificity = 1 - False Positive Rate. Changes dict key and value
                        if field_name == 'falsePositiveRateByLabel':
                            field_name = 'specificity'
                            value = 1 - value

                    # Adds metric/time name to dict if needed
                    if field_name not in n_runs[estimator_name]:
                        n_runs[estimator_name][field_name] = []

                    # Adds the value
                    n_runs[estimator_name][field_name].append(value)

        # Generates T5 and saves all the metrics and values in the CSV file
        for estimator_name in n_runs:
            all_t4_values = n_runs[estimator_name][T4_KEY]
            n_runs[estimator_name]['T5'] = f'{np.mean(all_t4_values)} (± {np.std(all_t4_values)})'

            res_dict = {
                'Model': estimator_name,
                'Hyper parameters description': '-',
                'Features': f'({n_features}) {features_names}',
                'T1': time_t1,
                'T2': time_t2
            }

            # Union with all the metrics and timing
            res_dict = dict(res_dict, **n_runs[estimator_name])

            # Adds to DataFrame
            result_csv = result_csv.append(res_dict, ignore_index=True)

        # Saves result to prevent data loss if an error occurs
        result_csv.to_csv(res_csv_file_path)
        exit()  # TODO: remove this to try all the n_features


if __name__ == '__main__':
    main()
