import csv
import argparse
import matplotlib.pyplot as plt
# from numpy import linspace
from glob import glob
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str)
parser.add_argument("--dataset", type=str, default='acdc')
parser.add_argument("--class_num", type=int, default=-1)
parser.add_argument("--idx", type=int, default=1)
args = parser.parse_args()

def return_at_index(idxs, vals, idx):
    all_dict = {idxs[i]: vals[i] for i in range(len(idxs))}
    return round(100*all_dict[idx], 1)

class_maps = {
0: 'road',
1: 'sidewalk',
2: 'building',
3: 'wall',
4: 'fence',
5: 'pole',
6: 'traffic light',
7: 'traffic sign',
8: 'vegetation',
9: 'terrain',
10:'sky',
11:'person',
12:'rider',
13:'car',
14:'truck',
15:'bus',
16:'train',
17:'motorcycle',
18:'bicycle', 
19: 'mIOU'
}


def main(args):
    base_path = './compressed_results/classes/'
    path_stream = base_path + class_maps[args.class_num]+'/' +  args.experiment
    path_stream_with_carla = base_path+ class_maps[args.class_num]+'/' +  args.experiment + '_with_carla'
    path_stream_with_viper = base_path+ class_maps[args.class_num]+'/' +  args.experiment + '_with_viper'
    path_stream_with_carla_and_viper = base_path+ class_maps[args.class_num]+'/' +  args.experiment + '_with_carla_and_viper'

    with open(path_stream, 'rb') as f:
        stream_results = pickle.load(f)
    with open(path_stream_with_carla, 'rb') as f:
        stream_results_with_carla = pickle.load(f)
    with open(path_stream_with_viper, 'rb') as f:
        stream_results_with_viper = pickle.load(f)
    # with open(path_stream_with_carla_and_viper, 'rb') as f:
    #     stream_results_with_carla_and_viper = pickle.load(f)

    legends = []
    x_stream, mIOUS_stream = zip(*stream_results[args.dataset])
    plt.scatter(x_stream, mIOUS_stream)
    legends += ['stream']

    x_stream_with_carla, mIOUS_stream_with_carla = zip(*stream_results_with_carla[args.dataset])
    plt.scatter(x_stream_with_carla, mIOUS_stream_with_carla)
    legends += ['stream_with_carla']

    x_stream_with_viper, mIOUS_stream_with_viper = zip(*stream_results_with_viper[args.dataset])
    plt.scatter(x_stream_with_viper, mIOUS_stream_with_viper)
    legends += ['stream_with_viper']

    # x_stream_with_carla_and_viper, mIOUS_stream_with_carla_and_viper = zip(*stream_results_with_carla_and_viper[args.dataset])
    # plt.scatter(x_stream_with_carla_and_viper, mIOUS_stream_with_carla_and_viper)
    # legends += ['stream_with_carla_and_viper']
    
    if class_maps[args.class_num] == 'mIOU':
        print("dataset ", args.dataset)
        print(return_at_index(x_stream, mIOUS_stream, args.idx))
        print(return_at_index(x_stream_with_carla, mIOUS_stream_with_carla, args.idx))
        print(return_at_index(x_stream_with_viper, mIOUS_stream_with_viper, args.idx))
        # print(return_at_index(x_stream_with_carla_and_viper, mIOUS_stream_with_carla_and_viper, 1))

        
    plt.grid()
    plt.legend(legends)
    plt.title(args.dataset + ' - ' + class_maps[args.class_num])
    plt.xlabel("# of steps")
    plt.ylabel('mIOU')

    per_class_path = './visualizations/'+ args.experiment + '/' + args.dataset
    import os
    os.makedirs(per_class_path, exist_ok=True)
    plt.savefig(per_class_path + '/' + class_maps[args.class_num] + '.png')
    plt.clf()
    return


if __name__ == '__main__':
    for dataset in ['cs', 'idd', 'bdd','acdc']:
        args.dataset = dataset
        for i in range(20):
            args.class_num = i
            # print("plotting for " + args.dataset + " and class " + class_maps[args.class_num])
            main(args)
    print("All good!")

