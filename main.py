# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  # Under `torchrun` multiple processes run this same script; only rank 0
  # should own the shared log file to avoid multiple writers clobbering it.
  rank = int(os.environ.get('RANK', 0))
  if FLAGS.mode == "train":
    # Create the working directory
    os.makedirs(FLAGS.workdir, exist_ok=True)
    logger = logging.getLogger()
    if rank == 0:
      # Set logger so that it outputs to both console and file
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
      handler = logging.StreamHandler(gfile_stream)
      formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
      handler.setFormatter(formatter)
      logger.addHandler(handler)
      logger.setLevel('INFO')
    else:
      logger.setLevel('WARNING')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
