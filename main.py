import json
from utils.utils import AnnotationAdjuster


if __name__=="__main__":
    
    imgPath="./images" #图片路径
    jsonPath="./demo.json"     #COCO语义分割注释文件
    with open(jsonPath,"r") as fp:
        train_anno=json.load(fp)
    Adjuster=AnnotationAdjuster(train_anno,imgPath,deleteLog="./deletetLog")
    Adjuster.adjustImg()
    # Adjuster.confirmDelete()
    # Adjuster.deleteFromJson()
    