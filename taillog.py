import time

while True:
  f = open("logs", 'r')
  print("READ: \n")
  print(len(f.readlines()))
  time.sleep(1)
