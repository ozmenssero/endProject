def calculateLesionLength(imageMatrix,lesionParameters):
  import numpy as np
  from skimage.filters import frangi,threshold_otsu,median
  from skimage.feature import corner_harris, corner_peaks
  from skimage import color,io
  from skimage.morphology import disk,square,closing
  # import matplotlib.pyplot as plt
  import cv2
  import math
  french=lesionParameters['french']
  pointCenters=lesionParameters['pointCenters']
  cropCoordinates=lesionParameters['cropCoordinates']
  mouseX,mouseY,lastX,lastY=cropCoordinates['mouseX'],cropCoordinates['mouseY'],cropCoordinates['lastX'],cropCoordinates['lastY']
  lesionLengthInPixel=round(lesionParameters['lesionLength'],5)
  print(cropCoordinates,lesionLengthInPixel)
  #img=cv2.imread("/content/angio4.jpeg")
  imgRGB=cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2RGB)
  imgGray = color.rgb2gray(imgRGB)
  # cv2_imshow(img)
  #io.imshow(frangied)

  #TOPHAT MORPH
  #filterSize =(3, 3)
  #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)
  kernel=np.ones((30,30),np.uint8)
  tophatImg=cv2.morphologyEx(imgGray,cv2.MORPH_BLACKHAT,kernel)
  # io.imshow(tophatImg,cmap=plt.cm.gray)

  #OTSU THRESH ???
  thresh=threshold_otsu(tophatImg)
  binary= tophatImg <thresh*0.6 #buyuktur kucuk degistir
  binary=binary.astype(float)   #binary image veya float yap burdan
  # io.imshow(binary,cmap=plt.cm.gray)

  #median
  med = median(binary, square(10))

  #FRANGI
  # fig, ax = plt.subplots(ncols=4,figsize=(15,15))

  frangieddd=frangi(image=med)

  # ax[0].imshow(img)
  # ax[0].set_title('Original image')

  # ax[1].imshow(frangied,cmap=plt.cm.gray)
  # ax[1].set_title('Frangi filter result')

  # ax[2].imshow(frangiedd,cmap=plt.cm.gray)
  # ax[2].set_title('Frangi filter result')

  # ax[3].imshow(frangieddd,cmap=plt.cm.gray)
  # ax[3].set_title('Frangi filter')
  # for a in ax:
  #     a.axis('off')

  # plt.tight_layout()
  # plt.show()

  thresh=threshold_otsu(frangieddd)
  sonq= frangieddd <thresh*0.4 #buyuktur kucuk degistir
  sonq=sonq.astype(float)   #binary image veya float yap burdan

  med2 = median(sonq, disk(7))

  from skimage import util
  from skimage.morphology import skeletonize
  # cv2_imshow(sonq)
  intbinar=med2.astype(float)  #BUNU DEGİSTİR SONUC İCİN
  intbinar=np.logical_not(intbinar).astype(float)

  # io.imshow(intbinar*255,cmap=plt.cm.gray)
  # intbinar=intbinar*255 # 0 and 1 to 0 and 255
  iskelet=skeletonize(intbinar)

  iskelet=iskelet.astype(np.uint8)
  # io.imshow(iskelet,cmap=plt.cm.gray)
  # cv2_imshow(sonq*255)
  # cv2_imshow(iskelet*255)
  # cv2_imshow(medials*255)

  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15),
  #                          sharex=True, sharey=True)
  # ax = axes.ravel()
  # ax[0].imshow(sonq, cmap=plt.cm.gray)
  # ax[0].axis('off')
  # ax[0].set_title('original', fontsize=20)

  # ax[1].imshow(iskelet, cmap=plt.cm.gray)
  # # ax[1].contour(sonq, [0.2], colors='r',linewidths=0.5)
  # ax[1].axis('off')
  # ax[1].set_title('skeleton', fontsize=20)



  # ax[2].imshow(dist_on_skel, cmap='magma')
  # # ax[2].contour(sonq, [0.2], colors='w',linewidths=0.5)
  # ax[2].set_title('medial_axis')
  # ax[2].axis('off')
  # fig.tight_layout()
  # plt.show()

  from skimage.measure import (label,regionprops,regionprops_table) #connected components uyguladim
  import pandas as pd 
  if(mouseX<lastX):
    temp=mouseX
    mouseX=lastX
    lastX=temp
  if(mouseY<lastY):
    temp=mouseY
    mouseY=lastY
    lastY=temp

  # cropped=iskelet[228:300,0:150]
  cropped=iskelet[lastY:mouseY,lastX:mouseX]
  labels = label(cropped,connectivity=2) #skimage doc'dan bak
  props=regionprops(labels)
  propsTable=regionprops_table(labels,properties=('label','area','coords','perimeter'))
  propsTable=pd.DataFrame(propsTable) 
  propsTable.sort_values(by='area',ascending=False,inplace=True)
  lesionLabel=propsTable.iloc[0]['label'] #lezyonun olduğu labeli buldum
  for i in props: #lezyonun olmadığı pikselleri 0 yapıyorum
    if(i.label!=lesionLabel):
      for j in i.coords:
        labels[j[0]][j[1]]=0
  # io.imshow(labels,cmap=plt.cm.gray)
  # propsTable.head()

  #CORNERLERI BUL
  cornerHarris=corner_harris(labels)
  cornerArray=corner_peaks(cornerHarris, min_distance=20, threshold_rel=0.07)

  if(len(cornerArray)):
    #3 POINT CENTER ILE BRANCH AYRIMI
    for a,b in cornerArray:#branch olan yerdeki 3x3 neighborlari sil
      neighbors=[[i,j] for i in range(a-1, a+2) for j in range(b-1, b+2) if i > -1 and j > -1 and j < len(labels[0]) and i < len(labels)]
      for c,d in neighbors:
        labels[c][d]=0

    labelhm=label(labels,connectivity=2)
    props2=regionprops(labelhm)
    propsTable2=regionprops_table(labelhm,properties=('label','area','coords','slice','perimeter'))
    propsTable2=pd.DataFrame(propsTable2) 
    coordsSeries=propsTable2['coords']
    
    for i in pointCenters:
      offsetX=i[0]-lastX
      offsetY=i[1]-lastY
      i[0]=offsetY
      i[1]=offsetX

    approximatedLabels=[]
    for point in pointCenters:
      labelAndMinDistance=[0,1000]
      for index,coordArray in coordsSeries.items(): #.loc index
        for coord in coordArray: #coord = [row, column]
          currentDistance=math.sqrt( ((point[0]-coord[0])**2)+((point[1]-coord[1])**2) )
          if(currentDistance<labelAndMinDistance[1]):
            labelAndMinDistance[0]=propsTable2.loc[index].label
            labelAndMinDistance[1]=currentDistance
      approximatedLabels.append(labelAndMinDistance[0])
    approximatedLabels=list(set(approximatedLabels))

    if(len(approximatedLabels)):
        for i in props2: #orientationun yakın olmadıgı labellari yani gereksiz branchleri silme
          if( i.label not in approximatedLabels ):
            for j in i.coords:
              labels[j[0]][j[1]]=0







  #son skeleton
  sonSkeleton=skeletonize(labels)
  sonSkeleton=sonSkeleton.astype(np.uint8)
  #________________________DEVAM
  #UZUNLUK BULMA DENEMESİ



  coords = np.column_stack(np.where(sonSkeleton ==1))
  stentLengthInPixel=0.0
  for i in range(len(coords)-1):
    euclideanDist = math.sqrt( ((coords[i][0]-coords[i+1][0])**2)+((coords[i][1]-coords[i+1][1])**2) )
    if(euclideanDist==1.0):
      stentLengthInPixel+=1.0
    elif(euclideanDist==math.sqrt(2)):
      stentLengthInPixel+=1.41421356
    else:
      continue
    
  if(len(cornerArray)): #neighbor ve cornerleri silindigi icin
    stentLengthInPixel+=3.0
  # return stentLengthInPixel
  if(french=="6"):
    catheterDiameterInMilimeter=0.2*10
  elif(french=="7"):
    catheterDiameterInMilimeter=0.2333*10
  
  stentLengthInMilimeter=(stentLengthInPixel*catheterDiameterInMilimeter)/lesionLengthInPixel
  return {'stentLengthInMilimeter':round(stentLengthInMilimeter,5),'stentLengthInPixel':round(stentLengthInPixel,5),
  'catheterDiameterInMilimeter':catheterDiameterInMilimeter,'catheterDiameterInPixel':lesionLengthInPixel}

  # import seaborn as sn
  # import pandas as pd
  # import matplotlib.pyplot as plt


  # df_cm = pd.DataFrame(cropped, range(cropped.shape[0]), range(cropped.shape[1]))
  # # plt.figure(figsize=(10,7))
  # sn.set(font_scale=1.4) # for label size
  # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

  # plt.show()
# calculateLesionLength("api\\bitirmedamar.png")