# Utility Functions
from nn import *
import numpy as np

powa = 100**2
def getRandomSample() -> tuple:
  # return np.random.randint(0,100) /100;
  # return (np.random.randint(0, 50) / 100, np.random.randint(0, 50) / 100);
  res = np.random.randint(0,powa) / powa
  assert res >= 0
  assert res <= 1
  return res
   # (np.random.randint(0, 50) / 100, np.random.randint(0, 50) / 100);

def getSampleLabel (sample):
    # return  sum(sample);
    sample = sample * powa
    return sample % 2 == 0 #sample[0] * sample[1]

# ANSI escape codes for colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# log_flag = True  # Set to False to disable logging
log_flag = False  # Set to False to disable logging
def log(message, color):
    global log_flag
    if log_flag:
        print(color + message + bcolors.ENDC)

logc_flag = False
# logc_flag = True
def logc(message):
    global logc_flag
    if logc_flag:
        print(message)

def showForwardProp(activations):
  print("--Showing Activations--")
  for actVec in activations:
      print(str(actVec.T))
      assert actVec.shape[1] == 1
  print("----Done----")


def showBackwardProp(gradient):
  logc ("---Computed Gradient---")
  logc ("---Printing Gradient Now")
  logc (str(gradient));
  logc ("---Finished Printing---");
  