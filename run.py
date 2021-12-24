import sys
import subprocess


mapping = {
    "rotate": "RotatE",
    "transe": "TransE"
}

if __name__ == "__main__":

    model = sys.argv[1].lower()
    dataset = sys.argv[2]
    gpu = sys.argv[3]
    options = sys.argv[4] if len(sys.argv > 4) else ""

    if model == "rotate":
        command = f"python codes/create_mapping.py data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16 -de"
    elif model == "transe":
        command = f"python codes/create_mapping.py data/{dataset} && " \
                    f"CUDA_VISIBLE_DEVICES={gpu} python -u codes/run.py --do_train --cuda --do_valid --do_test " \
                    f"--data_path data/{dataset} --model {mapping[model]} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                    f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save out/{dataset}_{model} " \
                    "--test_batch_size 16"

    if options == "dry-run":
        print(command)
    else:
        subprocess.run(command, shell=True)
