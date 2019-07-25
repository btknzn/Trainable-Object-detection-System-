# Trainable Object detection System:
 The aim of this system actually detecting line type for line following robot with camera but this can also apply other Image separating algorithm. It can use for handwriting separating, face separating and symbol separating. Or one camera application, you can specify object on camera specific area.

How does this code work?
   Firstly the user should specify  how many kind Image we are going to have and also have many sample ımage with camera, we are going to make. 
We are specifying the amount of different type Line
 
We are specifying the amount of sample Image of each type of Image.

After specifying amount, the user should make(press q in keyboard) Image with camera. When the user complete photographing part, system going to start learning  these Images. It is going to train with these images. This processes time depends on the amount of Image. After completing training part, the system is going  to want from user to make photo(press q in keyboard) with camera and It is going to specify the kind of this Image.

What is important about this code?

1.	The success of this code depends on main principle in Machine Learning. To be clear, More data is always better than better algorithm. Which means that when user increases the amount of Image and style of Image(distance, shadow on image, brightness of image, …)  , this system is going to work better.

How does this code work?
This code  is formed in 5 sub method and 1 main method.
Which are,
1.PreProcessing
2.FeatureDetection
3.takePictureFromVideo
4.datasetfromcamera
5.FindPicNumber
6.Main

Code  Description:
Firstly I am taking this ımage with noise  and not too good,
 

def preProcess(Img):
    img = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(img,(5,5),0)    
    ret,thr=cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
    laplacian = cv2.Laplacian(thr,cv2.CV_64F)
    return laplacian

   The reason why I convert my Image to Gray format is that. When I work with RGB image,  I am going to work with 3 dimension and which means I am going to spend 3 times more time because of this situation I am doing that.
  Why I am applying Gaussianblur and threshold  is for reducing Noise in Image. After applying Noise reduction, I applied my image in Lablacian, and I received boundary of my lines
 

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

This part for detecting feature of Image. Normally one image consists of 1024x1024x3 pixels. With converting this image to Gray format, we reduced to 1024x1024. In this part we are going to convert about one image to  (0-500)x2. In this way, our machine learning model is going to work  average 7864 times faster. And above  part I applied FastFeatureDetector algorithm to detect feature. You can see in below image. This algorithm just returns location of this feature. The below image has 124 features and af I applied this algorithm to detect  important features on this image.
 To summary, thanks to this algorithm, we do not need to seed our machine learning model with 1024x1024 data, we are going to seed with just 124 features.

 


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

Above function is for taking photo from video stream.

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


Above code for making dataset for our machine Learning system in one array like this. It is asking to user amount of image and how many kind of image is it going to be loaded.
1.column    =  X coordinate for feature
2.column   = Y coordinate for feature
3.column     =  which image is that (0 for 1. ımage , 1 for 2.ımage..)



def FindPicNumber(IndexPredict,i1):
    unique_elements, counts_elements = np.unique(IndexPredict, return_counts=True)
    value =0

    for i in range (len(counts_elements)-1):
        if counts_elements[i+1]>counts_elements[i]:
            value=i+1

    return unique_elements[value]


dataset,i1=dataSetFromCamera()

  The above code is finding  the most frequence image from the that which comes from our Machine learning system (MLP classifier)

def main()
	dataset,i1=dataSetFromCamera()
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
What main code makes


