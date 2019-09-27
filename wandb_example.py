import argparse
import fasttext
import os
import wandb

MODEL_NAME = ""
LR = 0.1
WORD_VECTOR_DIM = 100
EPOCHS = 10

# what would you want to log with a module like this?
# would want c++ bindings?
# i as a datascientist want to track the parameters that i AM tweaking

def train_fasttext(args):
   if args.model_name:
     os.environ["WANDB_DESCRIPTION"] = args.model_name
   wandb.init(project="fasttext")
   config = {
     "epochs" : args.epochs,
     "lr" : args.learning_rate,
     "word_vec_dim" : args.word_vector_dim
     
   } 
   wandb.config.update(config)
   model = fasttext.train_supervised(input="cooking.train", epoch=args.epochs, thread=1)

   #wandb.log({"loss" : model.loss})
#wandb.log({
#print(model.labels)
#model.save_model("model_cooking.bin")
#print(model.predict("I really like delicious vegetables but I'm not sure how to garnish them"))
#print(model.predict("what are your favorite healthy dessert recipes?"))
#print(model.predict("cucumbers melons apple pie cheese goat figs"))

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
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  
  train_args = parser.parse_args()
  # easier testing--don't log to wandb if dry run is set
  if train_args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  train_fasttext(train_args)

