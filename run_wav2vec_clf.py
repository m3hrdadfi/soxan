import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, load_metric
import numpy as np
import torch
import torchaudio

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    AutoConfig,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from src.collator import DataCollatorCTCWithPadding
from src.trainer import CTCTrainer

logger = logging.getLogger(__name__)
MODEL_MODES = ["wav2vec", "hubert"]
POOLING_MODES = ["mean", "sum", "max"]
DELIMITERS = {"tab": "\t", "comma": ",", "pipe": "|"}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_mode: str = field(
        default="wav2vec",
        metadata={
            "help": "Specifies the base model and must be from the following: " + ", ".join(MODEL_MODES)
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": "Specifies the reduction to apply to the output of Wav2Vec2 model and must be from the following: " + ", ".join(
                POOLING_MODES)
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained feature_extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to test on (a csv or JSON file)."},
    )
    input_column: Optional[str] = field(
        default="path",
        metadata={"help": "The name of the column in the datasets containing the audio path."},
    )
    target_column: Optional[str] = field(
        default="emotion",
        metadata={"help": "The name of the column in the datasets containing the labels."},
    )
    delimiter: Optional[str] = field(
        default="tab",
        metadata={
            "help": "Specifies the character delimiting individual cells in the CSV data and must be from the following: " + ", ".join(
                DELIMITERS.keys())
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples less than specified. Defaults to no filtering."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    logger.info(f"last_checkpoint: {last_checkpoint}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                    test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv",
            data_files=data_files,
            delimiter=DELIMITERS.get(data_args.delimiter, "\t"),
            cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    input_column_name = data_args.input_column
    output_column_name = data_args.target_column

    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features[output_column_name].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
        label_list = []
        logger.info(f"*** A regression problem ***")
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique(output_column_name)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

        logger.info(f"*** A classification problem with {num_labels} classes ***")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    setattr(config, 'pooling_mode', model_args.pooling_mode)

    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    # processor = Wav2Vec2Processor.from_pretrained(
    #     model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    target_sampling_rate = feature_extractor.feature_extractor.sampling_rate

    if model_args.model_mode == "wav2vec":
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_mode == "hubert":
        model = HubertForSpeechClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError("--model_mode does not exist in predefined modes: " + ",".join(MODEL_MODES))

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    # NOTE: Duration controller for the future `min_duration_in_seconds` `max_duration_in_seconds`
    # data_args.min_duration_in_seconds, data_args.max_duration_in_seconds

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def label_to_id(label, label_list):

        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1

        return label

    def preprocess_function(examples):
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column_name]]
        target_list = [label_to_id(label, label_list) for label in examples[output_column_name]]

        result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = datasets["train"]

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache
        )
        logger.info(f"Split sizes: {len(train_dataset)} train")

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = datasets["validation"]

        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache
        )
        logger.info(f"Split sizes: {len(eval_dataset)} validation")

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")

        predict_dataset = datasets["test"]

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache
        )
        logger.info(f"Split sizes: {len(predict_dataset)} test.")

    # Metric
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator
    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)

    # Initialize our Trainer
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=feature_extractor,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        logger.info(f"*** Training from: {checkpoint} ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Final test metrics
    if training_args.do_predict:
        logger.info("*** Test ***")

        predict_dataset.remove_columns_(output_column_name)
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w", encoding="utf-8") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    # NOTE: Pushing to hub for future
    # training_args.push_to_hub

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
