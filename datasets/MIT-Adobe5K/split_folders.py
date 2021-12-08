import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def split(eval_index, origin_path, train_root, eval_root, cla):
    images = os.listdir(origin_path)
    num = len(images)

    for index, image in enumerate(images):
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            image_path = os.path.join(origin_path, image)
            new_path = os.path.join(eval_root, cla)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            copy(image_path, new_path)
        else:
            # 将分配至训练集中的文件复制到相应目录
            image_path = os.path.join(origin_path, image)
            new_path = os.path.join(train_root, cla)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla.split("_")[0], index + 1, num), end="")  # processing bar
    print()


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向数据集文件夹
    data_root = "./data"
    low_dir_name = "dataA"
    high_dir_name = "dataB"

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)

    # 建立保存验证集的文件夹
    eval_root = os.path.join(data_root, "eval")
    mk_file(eval_root)

    origin_high_path = os.path.join(data_root, high_dir_name)
    assert os.path.exists(origin_high_path), "path '{}' does not exist.".format(origin_high_path)
    origin_low_path = os.path.join(data_root, low_dir_name)
    assert os.path.exists(origin_low_path), "path '{}' does not exist.".format(origin_low_path)

    high_images = os.listdir(origin_high_path)
    high_num = len(high_images)
    # 随机采样验证集的索引
    eval_index = random.sample(high_images, k=int(high_num * split_rate))

    split(eval_index, origin_low_path, train_root, eval_root, low_dir_name)
    split(eval_index, origin_high_path, train_root, eval_root, high_dir_name)

    print("processing done!")


if __name__ == '__main__':
    main()
