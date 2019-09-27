import argparse
import os
from subprocess import Popen

import time
import wandb

MODEL_NAME = ""
LR = 0.55
WORD_VECTOR_DIM = 100
EPOCHS = 25
THREADS = 10
NGRAMS = 1

logFileName = "logs"
lastLineId = 0
ft_cmd = "/usr/local/bin/fasttext supervised -input cooking.train -output tes -epoch {:d} -thread {:d} -lr {:04.2f} -wordNgrams {:d}"

def wandbLog():
  global lastLineId
  # log last values to wandb
  f = open(logFileName, 'r')
  lines = f.readlines()
  lenLog = len(lines)
  for l in lines[lastLineId:]:
    wandb.log({"loss" : float(l.rstrip())})
  lastLineId = lenLog  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=LR,
    help="learning rate")
  parser.add_argument(
    "-d",
    "--word_vector_dim",
    type=int,
    default=WORD_VECTOR_DIM,
    help="size of the word vectors")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of epochs")
  parser.add_argument(
    "-n",
    "--ngrams",
    type=int,
    default=NGRAMS,
    help="word n grams")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "-t",
    "--threads",
    type=int,
    default=THREADS,
    help="Number of CPU threads to run")
  
  args = parser.parse_args()
  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # wipe old logs
  if os.path.isfile(logFileName):
    os.remove(logFileName)
    f = open(logFileName, 'w')
    f.close()
   
  model_name_template = "{}lr{:03.2f} e{:d} t{:d}"
  run_name = model_name_template.format(args.model_name, args.learning_rate, args.epochs, args.threads)
  wandb.init(name=run_name, project="fasttext")
  config = {
    "epochs" : args.epochs,
    "lr" : args.learning_rate,
    "threads" : args.threads,
    "ngrams" : args.ngrams,
    "word_vec_dim" : args.word_vector_dim
  }
  wandb.config.update(config)
  ft_train_cmd = ft_cmd.format(args.epochs, args.threads, args.learning_rate, args.ngrams)
  print(ft_train_cmd)

  p = Popen(ft_train_cmd, shell=True)
  while p.poll() is None:
    wandbLog()
    time.sleep(1)
  wandbLog()

