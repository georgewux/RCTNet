import os
import cv2


def scale_width(image_high, image_low, target_width):
    """
    按照比例缩放图像
    :param image:
    :param target_width:
    :return scale_img:
    """
    ow, oh = image_high.shape[1], image_high.shape[0]

    if ow > oh:
        if ow == target_width:
            return image_high, image_low
        w = target_width
        h = int(target_width * oh / ow)
    else:
        if oh == target_width:
            return image_high, image_low
        w = int(target_width * ow / oh)
        h = target_width
    scale_high = cv2.resize(image_high, (w, h), cv2.INTER_CUBIC)
    scale_low = cv2.resize(image_low, (w, h), cv2.INTER_CUBIC)
    return scale_high, scale_low


def main():
    # 指向数据集文件夹
    data_root = "./"
    low_dir_name = "dataA"
    high_dir_name = "dataB"

    # 建立保存训练集的文件夹
    # train_root = os.path.join(data_root, "train")

    # 建立保存验证集的文件夹
    eval_root = os.path.join(data_root, "eval")

    # origin_high_path = os.path.join(train_root, high_dir_name)
    # assert os.path.exists(origin_high_path), "path '{}' does not exist.".format(origin_high_path)
    # origin_low_path = os.path.join(train_root, low_dir_name)
    # assert os.path.exists(origin_low_path), "path '{}' does not exist.".format(origin_low_path)

    origin_high_path = os.path.join(eval_root, high_dir_name)
    assert os.path.exists(origin_high_path), "path '{}' does not exist.".format(origin_high_path)
    origin_low_path = os.path.join(eval_root, low_dir_name)
    assert os.path.exists(origin_low_path), "path '{}' does not exist.".format(origin_low_path)

    high_images = os.listdir(origin_high_path)
    low_images = os.listdir(origin_low_path)

    for index, fname in enumerate(high_images):
        high_img_path = os.path.join(origin_high_path, fname)
        high_image = cv2.imread(high_img_path, 1)
        low_img_path = os.path.join(origin_low_path, fname)
        low_image = cv2.imread(low_img_path, 1)
        scale_high, scale_low = scale_width(high_image, low_image, 512)
        cv2.imwrite(high_img_path, scale_high)
        cv2.imwrite(low_img_path, scale_low)

    print("Processing done!")


if __name__ == '__main__':
    main()
