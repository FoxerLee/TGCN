import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='R8')
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
