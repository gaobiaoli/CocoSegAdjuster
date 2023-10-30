# **COCO语义分割注释文件调整工具**
* 交互式调整Coco语义分割的Category
* 支持**图片调整**和**实例调整**两种模式
## 1、图片模式
![](demo/ImgMode.png)
+ **鼠标左键单击**:跳过当前图片 
  
+ **鼠标右键单击**:调整鼠标位置的instance类别
  
+ **键盘小写“d”**:添加当前Img_id至deleteLog,用于后续的删除
  
+ **键盘“Esc”**:退出程序

```python
imgPath="./images" #图片路径
jsonPath="./demo.json"     #COCO语义分割注释文件
with open(jsonPath,"r") as fp:
    train_anno=json.load(fp)
Adjuster=AnnotationAdjuster(train_anno,imgPath,deleteLog="./deletetLog")
Adjuster.adjustImg()
```

## 2、实例模式
![](demo/InstanceMode.png)
+ **鼠标左键单击**:跳过当前instance

+ **鼠标右键单击**:调整当前instance

+ **鼠标右键双击**:调整当前图片所有instance

+ **鼠标滚轮单击**:跳过当前图片

+ **键盘小写“d”**:添加当前Img_id至deleteLog,用于后续的删除

+ **键盘“Esc”**:退出程序
  
```python
imgPath="./images" #图片路径
jsonPath="./demo.json"     #COCO语义分割注释文件
with open(jsonPath,"r") as fp:
    train_anno=json.load(fp)
Adjuster=AnnotationAdjuster(train_anno,imgPath,deleteLog="./deletetLog")
Adjuster.adjustInstance()
```

## 3、删除校对
确认log中的img_id是否需要删除，防止删错
```python
imgPath="./images" #图片路径
jsonPath="./demo.json"     #COCO语义分割注释文件
with open(jsonPath,"r") as fp:
    train_anno=json.load(fp)
Adjuster=AnnotationAdjuster(train_anno,imgPath,deleteLog="./deletetLog")
Adjuster.confirmDelete()
```
## 4、从Json中删除
```python
imgPath="./images" #图片路径
jsonPath="./demo.json"     #COCO语义分割注释文件
with open(jsonPath,"r") as fp:
    train_anno=json.load(fp)
Adjuster=AnnotationAdjuster(train_anno,imgPath,deleteLog="./deletetLog")
Adjuster.deleteFromJson()
```