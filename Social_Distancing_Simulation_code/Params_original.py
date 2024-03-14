


## some params related to exits
NEXITS = 5 # how many cashiers in the store
CASHIERD = 14 # distance between cashiers
ENTRANCEPOS = 0 # x-coord of entrance to the store
EXITPOS = 1 # Lx - y-coord of the first cashier 

## diffusion coeffs
DIFFCOEFF = 5e-2
#DIFFCOEFF = 0.0001
ACSINKCOEFF= 1e-2 # 1 / tau
PLUMEMIN=0.0

## plume spreading parameters
PLUMELIFETIME = 20 ## lifetime of plume for discrete plumes without diffusion
PLUMECONCINC = 40000.0 ## aerosol concentration in coughing event
PLUMECONCCONT = 5.0 ## continuous aerosol emission
PLUMEPLOTMIN=1 ## paramter for plotting method

CASHIERTIMEPERITEM = 1 ## waiting time multiplier on cashier
BLOCKRANDOMSTEP=0.8  ## parameter giving prob that customer takes random step if path blocked (i.e. another customer blocking)
PROBSPREADPLUME =  1./3600 # i.e. prob of cough per sec
### PROBSPREADPLUME =  1./60 # i.e. prob of cough per sec

## level 01 : >= 0.5
## level 02 : >= 1.0 (10^0)
## level 03 : >= 5.0 
## level 04 : >= 10 (10^1)
## level 05 : >= 50
EXPOSURELIMIT_LEVEL_01 = 0.5
EXPOSURELIMIT_LEVEL_02 = 1.0
EXPOSURELIMIT_LEVEL_03 = 5.0
EXPOSURELIMIT_LEVEL_04 = 10.0
EXPOSURELIMIT_LEVEL_05 = 50.0

EXPOSURELIMIT =1.0 ## a threshold for counting advanced exposure statistics 
##EXPOSURELIMIT=5.0 ## a threshold for counting advanced exposure statistics 
###EXPOSURELIMIT=10 ## a threshold for counting advanced exposure statistics 
##EXPOSURELIMITMAX=10 ## a threshold for counting advanced exposure statistics 
## limits for maximum waiting time when a target from the shopping list is found
MAXWAITINGTIME=2 
MINWAITINGTIME=1

## some simulation parameters for the customer behaviour
MAXSHOPPINGLIST=20 
WEIRDQUEUELIMIT = 39 ## parameter for starting plotting images when queues grow larger than the value


