import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--model', default='rotate', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--options', default='', help='Please provide additional instructions if necessary')

mapping = {
    "rotate": "RotatE",
    "transe": "TransE",
    "rotate_gate": "RotatE_Gate",
    "transe_gate": "TransE_Gate",
    "rotate_kbln": "RotatE_KBLN",
    "transe_kbln": "TransE_KBLN"
}

if __name__ == "__main__":

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    gpu = args.gpu
    options = args.options

    if model == "rotate":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 -de"
    elif model == "transe":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16"
    elif model == "rotate_gate":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"python codes/prepare_literals.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 -de --use_literal"
    elif model == "transe_gate":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"python codes/prepare_literals.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 --use_literal"
    elif model == "rotate_kbln":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"python codes/prepare_literals.py --dataset data/{dataset} && " \
                    f"python codes/preprocess_kg.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 -de --use_kbln"
    elif model == "transe_kbln":
        command = f"python codes/create_mapping.py --dataset data/{dataset} && " \
                    f"python codes/prepare_literals.py --dataset data/{dataset} && " \
                    f"python codes/preprocess_kg.py --dataset data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 --use_kbln"

    if options == "dry-run":
        print(command)
    else:
        subprocess.run(command, shell=True)
