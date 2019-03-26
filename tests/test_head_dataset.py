import unittest

from maskrcnn_benchmark.data.datasets import HeadDataset


def main():
    data_dir = "/workspace/csf/data/head_detect/train/yuncong_data/"
    gt_file = 'UCSD_train.txt'
    head = HeadDataset(data_dir, gt_file)
    from ipdb import set_trace
    set_trace()
    print('test')

if __name__ == '__main__':
    unittest.main()