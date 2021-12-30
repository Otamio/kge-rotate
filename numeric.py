import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--options', default='', help='Please provide additional instructions if necessary')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    gpu = args.gpu
    options = args.options

    command_transe = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                     f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                     f"--data_path data/{dataset} --model TransE -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                     f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_transe " \
                     "--test_batch_size 16 --use_stopper --save_best"

    command_rotate = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                     f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                     f"--data_path numeric/{dataset} --model RotatE -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                     f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save numeric/{dataset}_rotate " \
                     "--test_batch_size 16 -de --use_stopper --save_best"

    subprocess.run(command_transe, shell=True)
    subprocess.run(command_rotate, shell=True)
