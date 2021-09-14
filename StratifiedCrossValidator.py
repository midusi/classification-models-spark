import time
import numpy as np
from functools import reduce
from typing import List, Dict, Union, Tuple, cast
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.evaluation import JavaEvaluator
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.sql import DataFrame as SparkDataFrame

# Modified version of: https://github.com/interviewstreet/spark-stratifier

# Cross validation result consists in the metric or timing name, and the value. The least is, in case of
# timing, a list of values to get a mean, in case of metric it has a CrossValidationModel and the index of the
# best model
TimingList = List[float]
BestModelResult = Tuple[CrossValidatorModel, int]
CrossValidationResult = Dict[str, Dict[str, Union[TimingList, BestModelResult]]]

# Key in dict where every K fold timing is stored
K_TIMING_KEY = 'k_timing'


class StratifiedCrossValidator(CrossValidator):
    def stratify_data(self, dataset) -> List[SparkDataFrame]:
        """
        Returns an array of dataframes with the same ratio of passes and failures.
        Currently only supports binary classification problems.
        """
        nFolds = self.getOrDefault(self.numFolds)
        split_ratio = 1.0 / nFolds

        passes = dataset[dataset['label'] == 1]
        fails = dataset[dataset['label'] == 0]

        pass_splits = passes.randomSplit([split_ratio for _ in range(nFolds)])
        fail_splits = fails.randomSplit([split_ratio for _ in range(nFolds)])

        stratified_data = [pass_splits[i].unionAll(fail_splits[i]) for i in range(nFolds)]

        return stratified_data

    def _fit(self, dataset) -> CrossValidationResult:
        """
        Fitness function
        :param dataset: Dataset to compute cross validation
        :return: Stratified Cross Validator results for all the metrics and params
        """
        estimators: List[JavaEstimator] = self.getOrDefault(self.estimator)
        estimator_param_map = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(estimator_param_map)
        evaluators: List[JavaEvaluator] = self.getOrDefault(self.evaluator)
        numEvaluators = len(evaluators)
        nFolds = self.getOrDefault(self.numFolds)

        stratified_data = self.stratify_data(dataset)

        # best_models -> estimator_name -> evaluator_name -> best CrossValidator with best model index
        best_models: CrossValidationResult = {}

        # metrics_matrix has all the metrics obtained from all the possible models (specified in ParamGrid) for
        # every estimator and evaluator
        metrics_matrix = {}

        for k in range(nFolds):
            # Generates train and validation datasets
            train_arr = [x for j, x in enumerate(stratified_data) if j != k]
            train = reduce((lambda x, y: x.unionAll(y)), train_arr)
            validation = stratified_data[k]

            for estimator in estimators:
                estimator_name = estimator.__class__.__name__

                # If it's the first fold for the estimator, creates a List to collect K timing
                if not (estimator_name in best_models and K_TIMING_KEY in best_models[estimator_name]):
                    best_models[estimator_name] = {
                        K_TIMING_KEY: []
                    }

                # Adds the key of estimator name and sets a list for every evaluator which contains a list of metrics
                # obtained for every model (ParamGrid)
                if not (estimator_name in metrics_matrix):
                    metrics_matrix[estimator_name] = [[0.0] * numModels for _ in range(numEvaluators)]

                # Fits as many models as parameters in GridParam
                start_train = time.time()
                models = estimator.fit(train, estimator_param_map)
                time_t3 = time.time() - start_train
                best_models[estimator_name][K_TIMING_KEY].append(time_t3)

                # Evaluates using evaluators and GridParams and saves the obtained metrics
                for evaluator_idx, evaluator in enumerate(evaluators):
                    for model_idx in range(numModels):
                        model = models[model_idx]
                        predictions = model.transform(validation, estimator_param_map[model_idx])
                        metric = evaluator.evaluate(predictions)
                        metrics_matrix[estimator_name][evaluator_idx][model_idx] += metric / nFolds
                        print(f'Metrica {evaluator.getMetricName()} obtenida en el fold {k} '
                              f'para {estimator_name} (evaluator = {evaluator}) -> {metric}')

        # Analyses the best models for every estimator using all the specified evaluators
        for estimator in estimators:
            estimator_name = estimator.__class__.__name__
            for evaluator_idx, evaluator in enumerate(evaluators):
                all_models_avg_metrics = metrics_matrix[estimator_name][evaluator_idx]

                # Gets best model (from ParamGrid) for the current estimator and evaluator
                if evaluator.isLargerBetter():
                    best_model_index = np.argmax(all_models_avg_metrics)
                else:
                    best_model_index = np.argmin(all_models_avg_metrics)

                # Trains using best model from ParamGrid and saves in result dict
                best_model_index = cast(int, best_model_index)
                result = estimator.fit(dataset, estimator_param_map[best_model_index])

                best_cross_validator_model: CrossValidatorModel = self._copyValues(CrossValidatorModel(
                    result, all_models_avg_metrics
                ))
                best_models[estimator_name][evaluator.getMetricName()] = (best_cross_validator_model, best_model_index)

        return best_models
