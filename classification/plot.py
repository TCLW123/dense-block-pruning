#!/usr/bin/env python3

import argparse
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir',default= '../experiments/densenet40_cifar10/', type=str)
    parser.add_argument('--title',default= 'DenseNet40-baseline=94.42', type=str)
    parser.add_argument('--name', default=[
        # '319-supernet-0.1-[150, 225, 250]-pretrain_2021-12-02-22:46:17/',
        '351-sample-expected-lr=0.1_2021-12-20-22:34:39/',
        # '367-sample-expected-lr=0.1_2021-12-22-11:25:06/',
        '367-1-sample-expected-lr=0.1_2021-12-22-16:19:00/',
        # '366-sample-expected-lr=0.1_2021-12-22-11:22:57/',
        '366-1-sample-expected-lr=0.1_2021-12-22-16:21:15/',
        '365-sample-expected-lr=0.1_2021-12-22-11:21:54',
        '368-sample-expected-lr=0.1_2021-12-23-13:02:31/',
        '369-sample-expected-lr=0.1_2021-12-23-13:04:02/',
    ], type=str)
    parser.add_argument('--plot_num', default=10, type=int)
    # parser.add_argument('--name', default=['/densenet40_2021-11-06-14:34:36/','/densenet40-cut-longest_2021-11-06-14:52:53/'], type=str)
    args = parser.parse_args()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    # ax.grid()
    # fig = plt.figure(figsize=(12, 8), dpi=100)
    ax_color = ['royalblue','coral','m','limegreen','c','violet','greenyellow'
                ,'yellow', 'firebrick',
                'lightpink', 'orangered','deepskyblue','maroon']
    i = 0

    ### 父坐标系
    ax.set_title(args.title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    ### 插入子坐标系
    axins = inset_axes(ax, width="70%", height="50%", loc='center left',
                       bbox_to_anchor=(0.35, 0, 1, 1),
                       bbox_transform=ax.transAxes)

    xmin = 80
    xmax = 300
    ymin = 93.1
    ymax = 94.4
    tick = 0.1
    axins.set_ylim(ymin, ymax)
    ins_yticks = np.arange(ymin,ymax,tick)
    axins.set_yticks(ins_yticks)
    axins.set_xlim(xmin,xmax)

    for name in args.name:
        testP = os.path.join(args.expDir + name,'test.csv')
        testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)
        testI, testLoss, testErr = np.split(testData, [1,2], axis=1)
        if i > args.plot_num:
            ax.plot(testI, testErr, label=name, color=ax_color[i], marker='p')
        else:
            ax.plot(testI, testErr, label=name, color=ax_color[i])
        # ax.plot(testI, testErr, label=name, color=ax_color[i], marker='p')
        ax.legend(loc='lower right')

        ## 绘制子坐标系图
        if i > args.plot_num:
            axins.plot(testI, testErr, label=name, color=ax_color[i], marker='p')
        else:
            axins.plot(testI, testErr, label=name, color=ax_color[i])
        i += 1

        ### plot acc after cut
        # testC = os.path.join(args.expDir + name,'test_cut.csv')
        # testDataC = np.loadtxt(testC, delimiter=',').reshape(-1, 3)
        # testIC, testLossC, testErrC = np.split(testDataC, [1, 2], axis=1)
        # ax.plot(testIC, testErrC, label=name+'cut', color=ax_color[i])
        # ax.legend(loc='lower right')
        # # ax.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0)
        # axins.plot(testIC, testErrC, label=name+'cut', color=ax_color[i])
        # i += 1


    err_fname = os.path.join(args.expDir, 'acc.png')
    plt.show()
    plt.savefig(err_fname)

    print('Created {}'.format(err_fname))


def rolling(N, i, loss, err):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    loss_ = np.convolve(loss, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, loss_, err_

if __name__ == '__main__':
    main()
