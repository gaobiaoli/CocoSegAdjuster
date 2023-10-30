import cv2
import os
import json
import numpy as np
class AnnotationAdjuster:
    """
    COCO语义分割注释文件调整
    """
    def __init__(self,annoJson,imgPath,deleteLog="./deleteLog"):
        self.annoJson=annoJson
        self.imgPath=imgPath
        self.deleteImgId=[]
        self.deleteLog=deleteLog
        
    def _findImgInfoById(self,imgId):
        """根据instance_id获取Img_id"""
        for imgInfo in self.annoJson["images"]:
            if imgInfo["id"]==imgId:
                return imgInfo
            
    def _findInstanceIndexByImgId(self,imgId,start=0):
        '''根据ImgId查找其第一个Instance_index'''
        for index in range(start,len(self.annoJson['annotations'])):
            if self.annoJson['annotations'][index]['image_id']==imgId:
                return index
        
    def _nextImg(self,index):
        """根据instance_id获取下张图片的第一个instance_id"""
        if index>=len(self):
            return -1
        imgId=self.annoJson['annotations'][index]["image_id"]
        while index<len(self.annoJson['annotations']) and self.annoJson['annotations'][index]["image_id"]==imgId:
            index+=1
        return index
    
    def getImg(self,index):
        """根据Instance_Id获取包含全部anno的Img"""
        if index>=len(self):
            return None
        imgInfo=self._findImgInfoById(self.annoJson['annotations'][index]["image_id"])
        img=cv2.imread(os.path.join(self.imgPath,imgInfo['file_name']))
        if img is None:
            return None
        colorMask = np.zeros_like(img)
        self.indexMask = np.zeros(img.shape[0:2])
        lastIndex=self._nextImg(index)  #获取下张图片的第一个instance_index
        if lastIndex==-1:
            return None,None
        for i in range(index,lastIndex):
            anno=self.annoJson['annotations'][i]
            color=(0, 0, 255) if anno["category_id"]==0 else (255, 0, 255)
            seg=np.array(anno["segmentation"],dtype=np.int32).reshape(1,-1,2)
            cv2.fillPoly(colorMask, seg, color)
            cv2.fillPoly(self.indexMask, seg, i)
        result_image = cv2.addWeighted(img, 0.7, colorMask, 0.3, 0)
        print("当前进度：%d/%d"%(self.index,len(self.annoJson['annotations'])))
        return result_image

    def getImgByInstance(self,index):
        """根据Instance_Id获取包含当前anno的Img"""
        if index>=len(self):
            return None
        anno=self.annoJson['annotations'][index]
        imgInfo=self._findImgInfoById(anno["image_id"])
        img=cv2.imread(os.path.join(self.imgPath,imgInfo['file_name']))
        mask = np.zeros_like(img)
        seg=np.array(anno["segmentation"],dtype=np.int32).reshape(1,-1,2)
        color=(0, 0, 255) if anno["category_id"]==0 else (255, 0, 255)
        cv2.fillPoly(mask, seg, color)  
        result_image = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        return result_image

    def _adjust(self,instanceInfoDict):
        instanceInfoDict["category_id"]^=1

    def __len__(self):
        return len(self.annoJson["annotations"])
    def _setInstanceLabelCallBack(self,event, x, y, flags, param):
        """实例模式的回调函数"""
        if event==cv2.EVENT_LBUTTONDOWN: #左键点击，确认当前注释
            self.index+=1
        elif event==cv2.EVENT_RBUTTONDOWN: #右键点击，修改当前注释
            self._adjust(self.annoJson['annotations'][self.index])
            self.index+=1
        elif event==cv2.EVENT_MBUTTONDOWN: #滚轮点击，跳过当前图片
            self.index=self._nextImg(self.index)
        elif event==cv2.EVENT_RBUTTONDBLCLK: #右键双击，当前图片全部置为1
            tempindex=self._nextImg(self.index)
            for i in range(self.index,tempindex):
                self._adjust(self.annoJson['annotations'][i])
                self.index=tempindex
        else:
            return 
        print("当前进度：%d/%d"%(self.index,len(self.annoJson['annotations'])))
        result_img=self.getImgByInstance(self.index)
        if result_img is not None:
            cv2.imshow("img",result_img)
        else:
            cv2.destroyAllWindows()

    def _setImgCallBack(self,event, x, y, flags, param):
        """图片模式的回调函数"""
        if event==cv2.EVENT_LBUTTONDOWN: #左键点击，下一张图片
            self.index=self._nextImg(self.index)
            result_img=self.getImg(self.index)
            
        elif event==cv2.EVENT_RBUTTONDOWN: #右键点击，切换鼠标位置的category
            tempIndex=int(self.indexMask[y,x])
            self._adjust(self.annoJson['annotations'][tempIndex])
            result_img=self.getImg(self.index)
        else:
            return 
        if result_img is not None:
            cv2.imshow("img",result_img)
        else:
            cv2.destroyAllWindows()

    def adjustInstance(self,initIndex=0):
        """
            根据Instance逐个调整
            鼠标左键单击:跳过当前instance 
            鼠标右键单击:调整当前instance
            鼠标右键双击:调整当前图片所有instance
            鼠标滚轮单击:跳过当前图片
            键盘小写“d”:添加当前Img_id至deleteLog,用于后续的删除
            键盘“Esc”:退出程序
        """
        self.index=initIndex
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 1280, 960)
        
        while True:
            img=self.getImgByInstance(self.index)
            if img is None:
                break
            cv2.imshow("img",img)
            cv2.setMouseCallback("img", self._setInstanceLabelCallBack)
            key=cv2.waitKey(0)& 0xFF
            if key==27 or self.index>=len(self):
                break
            elif key==ord("d"):
                self.deleteImgId.append(self.annoJson['annotations'][self.index]['image_id'])
                self.index=self._nextImg(self.index)
        cv2.destroyAllWindows()
        self._autoSave()
    
    def adjustImg(self,initInstanceindex=0):
        """
            图片模式：逐个图片调整
            鼠标左键单击:跳过当前图片 
            鼠标右键单击:调整鼠标位置的instance类别
            键盘小写“d”:添加当前Img_id至deleteLog,用于后续的删除
            键盘“Esc”:退出程序
        """
        self.index=initInstanceindex
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 1280, 960)
        while True:
            if self.index>=len(self):
                break
            img=self.getImg(self.index)
            if img is None:
                break
            cv2.imshow("img",img)
            cv2.setMouseCallback("img", self._setImgCallBack)
            key=cv2.waitKey(0)& 0xFF
            if key==27 or self.index>=len(self):
                break
            elif key==ord("d"):
                self.deleteImgId.append(self.annoJson['annotations'][self.index]['image_id'])
                self.index=self._nextImg(self.index)
        cv2.destroyAllWindows()
        self._autoSave()

    def _autoSave(self):
        print("自动保存")
        if not os.path.exists(self.deleteLog):
            os.mkdir(self.deleteLog)
        historyCount=len(os.listdir(self.deleteLog))+1
        np.save(os.path.join(self.deleteLog,f"delete{historyCount}.npy"),self.deleteImgId)
        with open("adjust.json","w") as fp:
            json_str=json.dumps(self.annoJson,indent=4)
            fp.write(json_str)
    
    def confirmDelete(self,deleteFile=None):
        '''
        整合DeleteLog,逐个确认是否删除，防止删错
        优先读取deleteFile
        否则从DeleteLog目录中读
        '''
        if deleteFile is not None:
            deleteId=np.load(deleteFile,allow_pickle=True)
        else:
            deleteLogFile=os.listdir(self.deleteLog)
            deleteId=[]
            for log in deleteLogFile:
                deleteId+=np.load(os.path.join(self.deleteLog,log),allow_pickle=True).tolist()
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", 1280, 960)
        imgIndex=0
        confirmDelete=[]
        while imgIndex<len(deleteId):
            instanceIndex=self._findInstanceIndexByImgId(deleteId[imgIndex])
            img=self.getImg(instanceIndex)
            if img is None:
                break
            cv2.imshow("img",img)
            key=cv2.waitKey(0)& 0xFF
            if key==27:
                break
            elif key==ord("r"):
                imgIndex+=1
            elif key==ord("d"):
                confirmDelete.append(deleteId[imgIndex])
                imgIndex+=1
        np.save("./deleteConfirmed.npy",confirmDelete)

    def deleteFromJson(self,deleteFile="./deleteConfirmed.npy"):
        '''从json中删除'''
        deleteId=np.load(deleteFile,allow_pickle=True).tolist()
        self.annoJson['images']=[image for image in self.annoJson['images'] if image['id'] not in deleteId]
        self.annoJson['annotations']=[anno for anno in self.annoJson['annotations'] if anno['image_id'] not in deleteId]
        with open("deleted.json","w") as fp:
            json_str=json.dumps(self.annoJson,indent=4)
            fp.write(json_str)
    
    