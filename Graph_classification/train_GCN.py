#! /usr/bin/env python
from GCN import *
from datetime import datetime
from .utils.my_utils import *
from .utils.util import *
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import math
from train_utils import *


torch.manual_seed(124)
np.random.seed(124)

# Parameters
# ==================================================
parser = ArgumentParser("GCN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="Traceparts_6", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("The calculations will be performed on the device:", device)

# save paths
model_name = args.dataset + "_" + str(datetime.today().strftime('%m-%d-%h'))
out_dir = os.path.abspath(os.path.join(args.run_folder, "./result/runs_GCN", args.dataset))
if not os.path.exists(out_dir + "/Models/"):
    os.makedirs(out_dir + "/Models/")
save_path = out_dir + "/Models/" + model_name
print("Results will be saved in:", out_dir)
print("    The model will be saved as:", save_path)
print(args)

# Load data
# ==================================================
print("Loading data...")
use_degree_as_tag = False
graphs, num_classes = my_load_data(args.dataset, use_degree_as_tag)

train_graphs, test_graphs = separate_data(graphs, args.fold_idx)
train_graphs, valid_graphs = split_data(train_graphs, perc=0.9)
print("# training graphs: ", len(train_graphs))
print_data_commposition(train_graphs)
print("# validation graphs: ", len(valid_graphs))
print_data_commposition(valid_graphs)
print("# test graphs: ", len(test_graphs))
print_data_commposition(test_graphs)

feature_dim_size = graphs[0].node_features.shape[1]
print("Loading data... finished!")

# Model
# =============================================================
# Create a GCN model
model = GCN_CN_v4(feature_dim_size=feature_dim_size, num_classes=num_classes, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

# Main process
# =============================================================
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

train_losses = []
train_accuracy = []
valid_losses = []
valid_accuracy = []
valid_accuracy_x_class = []

best_loss = math.inf
best_accuracy = 0
# Train loop
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    # train model
    train(mmodel=model, optimizer=optimizer, train_graphs=train_graphs, batch_size=args.batch_size, num_classes=num_classes, device=device)
    # evaluate on train data
    train_loss, train_acc, _ = evaluate(mmodel=model, current_graphs=train_graphs, batch_size=args.batch_size, num_classes=num_classes, device=device, out_dir=out_dir)
    # evaluate on validation data
    valid_loss, valid_acc, valid_acc_x_class = evaluate(mmodel=model, current_graphs=valid_graphs, batch_size=args.batch_size, num_classes=num_classes, device=device, out_dir=out_dir)
    print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | valid acc {:5.2f} | '.format(epoch, (time.time() - epoch_start_time), train_loss, valid_loss, valid_acc*100))

    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracy.append(valid_acc)
    valid_accuracy_x_class.append(valid_acc_x_class)

    # Make a step of the optimizer if the mean of the last 6 epochs were better than the current epoch
    if epoch > 5 and train_losses[-1] > np.mean(train_losses[-6:-1]):
        scheduler.step()
        print("Scheduler step")
    # save if best performance ever
    if best_accuracy < valid_acc or (best_accuracy == valid_acc and best_loss > valid_loss):
        print("Save at epoch: {:3d} at valid loss: {:5.2f} and valid accuracy: {:5.2f}".format(epoch, valid_loss, valid_acc*100))
        best_accuracy = valid_acc
        best_loss = valid_loss
        torch.save(model.state_dict(), save_path)
    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(valid_acc*100) + '%\n')

# Plot results
# =============================================================
valid_accuracy_x_class = np.array(valid_accuracy_x_class).T
# plot training flow
plot_training_flow(ys=[train_losses, valid_losses], names=["train", "validation"], path=out_dir, fig_name="/losses_flow", y_axis="Loss")
plot_training_flow(ys=[np.array(train_accuracy)*100, np.array(valid_accuracy)*100], names=["train","validation"], path=out_dir, fig_name="/accuracy_flow", y_axis="Accuracy")
# Evaluate on test data
model.load_state_dict(torch.load(save_path))
test_loss, test_acc, _ = evaluate(mmodel=model, current_graphs=test_graphs, last_round=True)
print("Evaluate: loss on test: ", test_loss, " and accuracy: ", test_acc * 100)

write_acc.close()