

# class CONFIG(object):
#     """docstring for CONFIG"""
#     def __init__(self):
#         super(CONFIG, self).__init__()
        
#         self.dataset = 'R8'
#         self.model = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
#         self.learning_rate = 0.02   # Initial learning rate.
#         self.epochs  = 200  # Number of epochs to train.
#         self.hidden1 = 200  # Number of units in hidden layer 1.
#         self.dropout = 0.5  # Dropout rate (1 - keep probability).
#         self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
#         self.early_stopping = 10 # Tolerance for early stopping (# of epochs).
#         self.max_degree = 3      # Maximum Chebyshev polynomial degree.


import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', default=0.02)
args.add_argument('--epochs', default=2000)
args.add_argument('--hidden1', default=200)
args.add_argument('--dropout', default=0.5)
args.add_argument('--weight_decay', default=0.)
args.add_argument('--early_stopping', default=10)
args.add_argument('--max_degree', default=3)


args = args.parse_args()
print(args)
