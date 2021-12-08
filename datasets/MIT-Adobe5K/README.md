## Preparing data for the MIT Adobe FiveK Dataset with Lightroom

### Getting the data

* Download the dataset from https://data.csail.mit.edu/graphics/fivek/. (The "single archive (~50GB, SHA1)" or "by parts", either is fine).
* Extract the data
* Open the file ```fivek.lrcat```. Lightroom may probably ask you to upgrade. Just click "upgrade" if you are asked to. You may need to wait for a while (~10 minutes).

### Generating the Training Input Set

* Open the dataset with Adobe Lightroom.
* In the ```Collections``` list, select collection ```Inputs/(default) Input with ExpertC WhiteBalance minus1.5```.
* Select all images in the bottom (select one and press ```Ctrl-A```), right-click on any of them, choose ```Export/Export...```
    - Export Location: ```Export to```=```Specific folder```, 

      ```Folder```=```datasets/MIT-Adobe5K/Five_K/dataA/```.
    - File Naming: click ```Rename To```, and select ```Edit...```. Clear the edit box at first. Then, choose```{Sequence # (0001)>>}``` in ```Sequence and Date```, and click ```Insert```.
    - ```Image Format```=```JPEG``` (using ```jpg``` should be fine). ```Quality```=100. ```Color Space```=```sRGB```
    - Image Sizing: ```Resize to Fit```=```Long Edge```. Click ```Don't Enlarge```. Fill in ```500``` ```pixels```. ```Resolution``` doesn't matter since it is not the actual image resolution in pixels.
    - Finally, click ```Export```.
* You can compare the first exported image (0001.jpg) with [this image](https://github.com/MagicGeorge/RCTNet/raw/master/datasets/MIT-Adobe5K/input.jpg). If you have done the previous steps correctly, you should get an identical image.

### Generating the Training Target Set (Expert C)

* Open the dataset with Adobe Lightroom.
* In the ```Collections``` list, select collection ```Experts/C```.
* Select all images in the bottom (select one and press ```Ctrl-A```), right-click on any of them, choose ```Export/Export...```
    - Export Location: ```Export to```=```Specific folder```, 

      ```Folder```=```datasets/MIT-Adobe5K/Five_K/dataB/```.
    - File Naming: click ```Rename To```, and select ```Edit...```. Clear the edit box at first. Then, choose```{Sequence # (0001)>>}``` in ```Sequence and Date```, and click ```Insert```.
    - ```Image Format```=```JPEG```. ```Quality```=100. ```Color Space```=```sRGB```
    - Image Sizing: ```Resize to Fit```=```Long Edge```. Click ```Don't Enlarge```. Fill in ```500``` ```pixels```. ```Resolution``` doesn't matter since it is not the actual image resolution in pixels.
    - Finally, click ```Export```.
* You can compare the first exported image (0001.jpg) with [this image](https://github.com/MagicGeorge/RCTNet/raw/master/datasets/MIT-Adobe5K/target.jpg). If you have done the previous steps correctly, you should get an identical image.
