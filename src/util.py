# Utility Functions
from nn import *

def getRandomSample() -> tuple:
  #return np.random.randint(0,100) /100;
  return (np.random.randint(0, 50) / 100, np.random.randint(0, 50) / 100);

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

log_flag = True  # Set to False to disable logging
def log(message, color):
    global log_flag
    if log_flag:
        print(color + message + bcolors.ENDC)

