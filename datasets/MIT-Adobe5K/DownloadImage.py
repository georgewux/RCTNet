import os
import sys
from urllib.request import urlretrieve

# 改变当前工作路径
CURRENT_PATH = "E:\\WuWeiran\\RCTNet\\datasets\\MIT-Adobe5K"  # 本文件所在路径
os.chdir(CURRENT_PATH)  # 改变当前路径

# 存储图像名称的list
img_lst = []
# 读取图片名列表
with open('filesAdobe.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n"))  # 去掉换行符

with open('filesAdobeMIT.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n"))  # 去掉换行符


# urlretrieve 函数的回调函数，显示下载进度
def cbk(a, b, c):
    '''回调函数
    @a:已经下载的数据包数量
    @b:数据块的大小
    @c:远程文件的大小
    '''
    per = 100.0 * a * b / c
    if per > 100:
        per = 100
    # 在终端更新进度
    sys.stdout.write("progress: %.2f%%   \r" % per)
    sys.stdout.flush()


# 根据文件的url下载图片
for i in img_lst:
    # 下载由C所调整的图像(可根据需要下载其它的四类图像)下面的链接中dng改为tiff16_c
    URL = 'https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/' + i + '.tif'
    print('Downloading ' + i + ':')
    # 将所获取的图片存储到本地的地址内
    urlretrieve(URL, 'E:\\WuWeiran\\RCTNet\\datasets\\MIT-Adobe5K\\ExpertC\\' + i + '.tif', cbk)
