import argparse
from argparse import Namespace

parser = argparse.ArgumentParser()

parser.add_argument('square', help='Squares a given number', type=int, default=0)
parser.add_argument('-v', '--verbose', help='Provides a verbose desc', 
                    #action='store_true'
                    type=int,
                    choices=[0, 1, 2]
                    )


args: Namespace = parser.parse_args()
if args.verbose == 0:
    print('Option 1')
elif args.verbose == 1:
    print('Option 2')
elif args.verbose == 2:
    print('Option 3')    
# print(args.square**2)
