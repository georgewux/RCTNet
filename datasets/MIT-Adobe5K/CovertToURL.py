# 将生成的new.txt中的网址全选粘贴到迅雷中下载

_list = []
with open('filesAdobe.txt', 'r') as f:
    for line in f:
        _list.append("https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/" + line.replace("\n", "") + ".tif")
f.close()
with open('filesAdobeMIT.txt', 'r') as f:
    for line in f:
        _list.append("https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/" + line.replace("\n", "") + ".tif")
f.close()
f = open('new.txt', 'w')    # 若是'wb'就表示写二进制文件
for i in _list:
    f.write(i)
    f.write('\n')
f.close()
