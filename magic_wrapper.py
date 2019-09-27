from subprocess import Popen

commands = ["python taillog.py", "python wandb_example.py -m test_sub -q"]


processes = [Popen(cmd, shell=True) for cmd in commands]
# do other things here..
# wait for completion
for p in processes: 
  p.wait()
