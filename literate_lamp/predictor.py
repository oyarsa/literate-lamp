"Predictor for the McScript dataset"
from typing import TextIO, cast, Sequence

from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.training.metrics import CategoricalAccuracy


@Predictor.register('mcscript-predictor')
class McScriptPredictor(Predictor):
    """
    Predictor for our Question Answering task. It takes a `Model` and a
    `DatasetReader`, from which it can read a dataset and compute predictions
    on it using the model.
    """

    def predict(self,
                passage_id: str,
                question_id: str,
                question_type: str,
                passage: str,
                question: str,
                answer0: str,
                answer1: str) -> JsonDict:
        """
        Takes the sample data and creates a prediction JSON from it.
        This will then be used to create an `Instance` that will be passed to
        the model.
        """
        return self.predict_json({
            "passage_id": passage_id,
            "question_id": question_id,
            "question_type": question_type,
            "passage": passage,
            "question": question,
            "answer0": answer0,
            "answer1": answer1
        })

    # Extracts data from a JSON and creates an `Instance`.
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        passage_id = json_dict['passage_id']
        question_id = json_dict['question_id']
        question_type = json_dict['question_type']
        passage = json_dict['passage']
        question = json_dict['question']
        answer0 = json_dict['answer0']
        answer1 = json_dict['answer1']

        return self._dataset_reader.text_to_instance(
            passage_id, question_id, question_type, passage,
            question, answer0, answer1
        )


def score_questions(model: Model,
                    output_file: TextIO,
                    testset: Sequence[Instance]
                    ) -> float:
    "Computes the answers for each passage-question pair."
    metric = CategoricalAccuracy()

    for instance in testset:
        prediction = model.forward_on_instance(instance)

        correct = instance['label']
        predicted = prediction['prob'].argmax()

        metric(prediction['prob'], correct)

        passage_id = instance['passage_id']
        question_id = instance['question_id']

        print('{},{},{}'.format(passage_id, question_id, predicted),
              file=output_file)

    return cast(float, metric.get_metric())
