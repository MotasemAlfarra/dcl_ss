import csv
import os
import argparse
import matplotlib.pyplot as plt
# from numpy import linspace
from glob import glob
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str)
parser.add_argument("--dataset", type=str, default='acdc')
parser.add_argument("--class_num", type=int, default=-1)

args = parser.parse_args()


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
    base_path = './results/' + args.experiment 
    mIOUs_acdc, mIOU_cs, mIOU_bdd, mIOU_idd = [], [], [], []
    x_acdc, x_cs, x_bdd, x_idd = [], [], [], []
    for dataset in ['acdc', 'cityscapes', 'bdd', 'idd']:
        args.dataset = dataset
        complementary_path = base_path +'/*' + args.dataset + '.csv'
        print("plotting for " + args.dataset + " and class " + class_maps[args.class_num])
        for path in glob(complementary_path):
            x = (path.split('/')[-1]).split('_')
            # print(path)
            with open(path) as f:
                r = csv.reader(f)
                for row in r:
                    if args.dataset == 'bdd':
                        mIOU_bdd.append(float(row[args.class_num+1]))
                        x_bdd.append(int(x[-5]))
                    elif args.dataset == 'cityscapes':
                        mIOU_cs.append(float(row[args.class_num+1]))
                        x_cs.append(int(x[-5]))
                    elif args.dataset =='acdc':
                        mIOUs_acdc.append(float(row[args.class_num+1]))
                        x_acdc.append(int(x[-5]))
                    elif args.dataset =='idd':
                        mIOU_idd.append(float(row[args.class_num+1]))
                        x_idd.append(int(x[-5]))
                    else:
                        print("Fuck Off!")
    Legends= ['ACDC', 'CS', 'BDD', 'IDD']
    plt.scatter(x_acdc, mIOUs_acdc, c='g')
    plt.scatter(x_cs, mIOU_cs, c='b')
    plt.scatter(x_bdd, mIOU_bdd, c='r')
    plt.scatter(x_idd, mIOU_idd, c='purple')
    plt.grid()
    plt.legend(Legends)
    plt.title(args.experiment + ' - ' + class_maps[args.class_num])
    plt.xlabel("# of steps")
    plt.ylabel('mIOU')

    visualization_path = './visualizations/classes/' + class_maps[args.class_num]
    os.makedirs(visualization_path, exist_ok=True)
    plt.savefig(visualization_path + '/' +args.experiment+ '.png')
    plt.clf()

    results_acdc = zip(x_acdc, mIOUs_acdc)
    results_cs = zip(x_cs, mIOU_cs)
    results_bdd = zip(x_bdd, mIOU_bdd)
    results_idd = zip(x_idd, mIOU_idd)

    all_results = {
        'acdc': results_acdc,
        'cs': results_cs,
        'bdd': results_bdd,
        'idd': results_idd
    }

    save_path = './compressed_results/classes/' + class_maps[args.class_num]
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + '/' + args.experiment, 'wb') as f:
        pickle.dump(all_results, f)

    return


if __name__ == '__main__':
    for i in range(20):
        args.class_num = i
        main(args)
    print("All good!")

