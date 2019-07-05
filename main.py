import tensorflow as tf
from RCA_net import RCA_net
from mode import *
import argparse

parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')

## Model specification
parser.add_argument("--channel", type = int, default = 3)
parser.add_argument("--scale", type = int, default = 2)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--n_RG", type = int, default = 10)
parser.add_argument("--n_RCAB", type = int, default = 20)
parser.add_argument("--kernel_size", type = int, default = 3)
parser.add_argument("--ratio", type = int, default = 16)

## Data specification 
parser.add_argument("--train_GT_path", type = str, default = "./HR")
parser.add_argument("--train_LR_path", type = str, default = "./LR")
parser.add_argument("--test_GT_path", type = str, default = "./val_HR")
parser.add_argument("--test_LR_path", type = str, default = "./val_LR")
parser.add_argument("--patch_size", type = int, default = 48)
parser.add_argument("--result_path", type = str, default = "result")
parser.add_argument("--model_path", type = str, default = "./model")
parser.add_argument("--in_memory", type = str2bool, default = True)


## Optimization
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--max_step", type = int, default = 1 * 1e6)
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--decay_step", type = int, default = 2 * 1e5)
parser.add_argument("--decay_rate", type = float, default = 0.5)
parser.add_argument("--test_with_train", type = str2bool, default = False)
parser.add_argument("--save_test_result", type = str2bool, default = False)

## Training or test specification
parser.add_argument("--mode", type = str, default = "train")
parser.add_argument("--fine_tuning", type = str2bool, default = False)
parser.add_argument("--load_tail_part", type = str2bool, default = True)
parser.add_argument("--log_freq", type = int, default = 1e4)
parser.add_argument("--model_save_freq", type = int, default = 2 * 1e5)
parser.add_argument("--pre_trained_model", type = str, default = "./")
parser.add_argument("--self_ensemble", type = str2bool, default = False)
parser.add_argument("--chop_forward", type = str2bool, default = False)
parser.add_argument("--chop_shave", type = int, default = 10)
parser.add_argument("--chop_size", type = int, default = 4 * 1e4)
parser.add_argument("--test_batch", type = int, default = 1)
parser.add_argument("--test_set", type = str, default = 'benchmark')


args = parser.parse_args()

model = RCA_net(args)
model.build_graph()

print("Build model!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

if args.mode == 'train':
    train(args, model, sess)
    
elif args.mode == 'test':
    test(args, model, sess)
        
elif args.mode == 'test_only':
    test_only(args, model, sess)
