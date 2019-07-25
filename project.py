import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier

def preProcess(Img):
    img = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(img,(5,5),0)    
    ret,thr=cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
    laplacian = cv2.Laplacian(thr,cv2.CV_64F)
    return laplacian


def featureDetecition(img):
    img = img/255
    img = img.astype(np.uint8)
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img,None)
    index = []
    for point in kp:
        temp =list(point.pt)
        index.append(temp)
    index = np.array(index)
    return index



def takePictureFromVideo():
    video = cv2.VideoCapture(0)
    while True:
        check,frame = video.read()
        cv2.imshow("Capturing",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return frame


def dataSetFromCamera():
    i1 = int(input("İnput how many different type plaque there is"))
    i2 = int(input("input how many train fotograph do you want to take"))
    dataset = np.zeros((90000000,3))
    startPoint = 0
    for i in range(i1):

        for x in range (i2):
            Image = takePictureFromVideo()
            Image = preProcess(Image)
            Index = featureDetecition(Image)
            dataset[startPoint:startPoint+len(Index),2] = dataset[startPoint:startPoint+len(Index),2]+i
            dataset[startPoint:startPoint+len(Index),0] = Index[0:len(Index),0]
            dataset[startPoint:startPoint + len(Index), 1] = Index[0:len(Index), 1]
            startPoint=startPoint+len(Index)
        print("one picture complete")
    return dataset[0:startPoint],i1


def FindPicNumber(IndexPredict,i1):
    unique_elements, counts_elements = np.unique(IndexPredict, return_counts=True)
    value =0

    for i in range (len(counts_elements)-1):
        if counts_elements[i+1]>counts_elements[i]:
            value=i+1

    return unique_elements[value]


dataset,i1=dataSetFromCamera()

def main()

	mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
	mlp.fit(dataset[:,0:2],dataset[:,2])
	while (True):
		Image = takePictureFromVideo()
		Image = preProcess(Image)
		Index = featureDetecition(Image)
		IndexPredict=mlp.predict(Index)

		ImageNumber=FindPicNumber(IndexPredict,i1)
		print("Number of Image:")
		print(ImageNumber)



	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
	
main()