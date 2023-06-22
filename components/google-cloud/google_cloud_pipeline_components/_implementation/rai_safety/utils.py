"""Util functions for Vertex LLM evaluation pipelines."""

import argparse
import os
import random
import string
import tempfile
from typing import Any

from google.cloud import aiplatform
from kfp.v2.compiler import Compiler


def parse_args(title: str, args: list[str]):
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(prog=title)
  parser.add_argument(
      '--pipeline',
      dest='pipeline',
      type=str,
      default='llm_eval_minimal_pipeline',
      help=(
          'String of pipeline to run. Defaulted to '
          ' "llm_eval_minimal_pipeline".'
      ),
  )
  parser.add_argument(
      '--project', dest='project', type=str, default='model-evaluation-dev'
  )
  parser.add_argument(
      '--location', dest='location', type=str, default='us-central1'
  )
  parser.add_argument(
      '--root_dir',
      dest='root_dir',
      type=str,
      default='gs://model-evaluation-test-data/pipeline_root',
  )
  parser.add_argument(
      '--evaluation_task',
      dest='evaluation_task',
      type=str,
      default='text-generation',
  )
  parser.add_argument('--model_name', dest='model_name', type=str, default='')
  parser.add_argument(
      '--joined_predictions_gcs_source',
      dest='joined_predictions_gcs_source',
      type=str,
      default='',
  )

  parser.add_argument(
      '--predictions_gcs_source',
      dest='predictions_gcs_source',
      type=str,
      default='',
  )
  parser.add_argument(
      '--ground_truth_gcs_source',
      dest='ground_truth_gcs_source',
      type=str,
      default='',
  )
  parser.add_argument(
      '--batch_predict_gcs_destination_output_uri',
      dest='batch_predict_gcs_destination_output_uri',
      type=str,
      default='',
  )
  parser.add_argument(
      '--slice_spec_gcs_source',
      dest='slice_spec_data_path',
      type=str,
      default='',
  )

  # Kokoro Testing Parameters
  parser.add_argument(
      '--kokoro_test',
      dest='kokoro_test',
      default=False,
      action='store_true',
      help=(
          'If true, override project root directory and id to'
          ' "model-evaluation-e2e" project for Kokoro testing.'
      ),
  )
  parser.add_argument(
      '--wait',
      dest='wait',
      default=False,
      action='store_true',
      help='If true, block blaze on jobs to complete.',
  )
  parsed_args, _ = parser.parse_known_args(args)
  return parsed_args


def get_parameters_from_input_args_for_pipeline(
    parsed_args, pipeline_func
) -> dict[str, Any]:
  """Gets parameters based on pipeline and preexisting model if requested."""
  parsed_args_dict = vars(parsed_args)
  parameters = {}
  pipeline_args = list(
      pipeline_func.pipeline_spec.root.input_definitions.parameters.keys()
  )
  for arg in pipeline_args:
    if arg in parsed_args_dict:
      parameters[arg] = parsed_args_dict[arg]
  return parameters


def _random_str() -> str:
  """Generates a random 8 character string for better readability."""
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))


def run_pipeline(
    pipeline,
    parameters: dict[str, Any],
    project: str = 'model-evaluation-dev',
    location: str = 'us-central1',
    pipeline_root: str = 'gs://model-evaluation-test-data-us-central1/pipeline_root',
) -> aiplatform.PipelineJob:
  """Launches KFP pipeline on a test cluster."""

  tmp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
  Compiler().compile(
      pipeline_func=pipeline, package_path=tmp_file.name, type_check=True
  )
  aiplatform.init(
      project=project,
      location=location,
  )

  test_prefix = str(os.environ.get('USER', 'test'))
  test_name = (
      f'{test_prefix}-{str(pipeline.name).replace("_", "-")}-{_random_str()}'
  )
  job = aiplatform.PipelineJob(
      display_name=test_name,
      template_path=tmp_file.name,
      job_id=test_name,
      pipeline_root=pipeline_root,
      parameter_values=parameters,
      enable_caching=os.environ.get('ENABLE_CACHE', True),
  )

  job.submit()

  return job
