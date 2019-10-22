## UPSNet(Optional) 


### Requirements:

easydict:
```
pip install easydict
```
panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
cocoapi:
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext inst
```


For implementing UPSNet we are going to use the COCOval2017 dataset. 

**Before  runing  any `.sh` file, you should do this:**

  - Make the script executable with command `chmod +x <fileName>`.
  - Run the script using `./<fileName>.`, i.e., the full path to the `.sh` file.

Once you have clone the repository run `init.sh` to build essential C++/CUDA modules which also download the pretained model.


### COCO datset and annotations 

Download the following files:

[COCOval2017]
(http://images.cocodataset.org/zips/val2017.zip) (1GB)

[Train/val annotations]
(http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (241MB)

[Stuff train/val annotations]
(http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip) (1.1GB)

[Panoptic Train/Val annotations]
(http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) (821MB)


Once you have download all the files, create a folder named $COCO_ROOT which have  `annotations` and `images` folders under it.

Create a soft link by `ln -s $COCO_ROOT data/coco` under your `UPSNet current folder`.


If you don´t understand how does a softlink works check: 
https://www.cyberciti.biz/faq/creating-soft-link-or-symbolic-link/ 


Now you can check which files are under the new soft link runing
`$ ls -l  $COCO_ROOT data/coco`

Run `init_coco.sh` to prepare COCO dataset for UPSNet.

Run `download_weights.sh` to get trained model weights for COCO.


### Requirements:

 - To get the PQ results easier, you should replace the file that we are giving to you, `base_dataset.py`, to the folder `upsnet/dataset/`. The difference is that this new file would create a `result.json` were you can find the PQ metrics. 
 - **You need to change the `.yaml` file which is located in the `experiments` folder depending on how many GPUs you are using, and its ID.** 

### Test the model:
Finally, you can test the model in the  COCO validation dataset and obtain the qualitative and quatitavie results runing: 

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml --weight_path model/upsnet_resnet_50_coco_90000.pth
```



Once the test phase has finished you will find a folder called `output`.

Under that folder you will find the Panoptic Quality results in :

`output/upsnet/coco/upsnet_resnet50_coco_4gpu/val2017/results/pans_unified/results.json` 

At the  begining of the `results.json` file you will find the Panoptic Quality for all the classes of the COCO val dataset. At the end of it you will find the Panoptic quality for Things and Stuff classes. 


For the qualitative results you will find the images under the folder `pan` which you can find in:

`output/upsnet/coco/upsnet_resnet50_coco_4gpu/val2017/results/pans_unified/pan`






