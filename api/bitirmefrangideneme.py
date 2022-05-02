def calculateLesionLength(imageMatrix,lesionParameters):
  import numpy as np
  from skimage.filters import frangi,threshold_otsu
  from skimage import color,io
  # import matplotlib.pyplot as plt
  import cv2

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
  kernel=np.ones((20,20),np.uint8)
  tophatImg=cv2.morphologyEx(imgGray,cv2.MORPH_BLACKHAT,kernel)
  # io.imshow(tophatImg,cmap=plt.cm.gray)

  #OTSU THRESH ???
  thresh=threshold_otsu(tophatImg)
  binary= tophatImg <thresh*1.2 #buyuktur kucuk degistir
  binary=binary.astype(float)   #binary image veya float yap burdan
  # io.imshow(binary,cmap=plt.cm.gray)

  #FRANGI
  # fig, ax = plt.subplots(ncols=4,figsize=(15,15))

  frangied=frangi(image=imgGray,alpha=2,beta=2,gamma=15)
  frangiedd=frangi(image=tophatImg,black_ridges=False)
  frangieddd=frangi(image=binary)

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
  sonq= frangieddd <thresh #buyuktur kucuk degistir
  sonq=sonq.astype(float)   #binary image veya float yap burdan
  # io.imshow(sonq,cmap=plt.cm.gray)
  ikinci=frangi(image=sonq,alpha=2,beta=2,gamma=15)
  # io.imshow(ikinci,cmap=plt.cm.gray)

  from skimage import util
  from skimage.morphology import skeletonize,medial_axis,thin #thin'e de bak ise yarayabilir
  # cv2_imshow(sonq)
  intbinar=sonq.astype(float)  #BUNU DEGİSTİR SONUC İCİN
  intbinar=np.logical_not(intbinar).astype(float)

  # io.imshow(intbinar*255,cmap=plt.cm.gray)
  # intbinar=intbinar*255 # 0 and 1 to 0 and 255
  iskelet=skeletonize(intbinar)
  medials,misal=medial_axis(intbinar,return_distance=True)

  dist_on_skel=medials*misal
  iskelet=iskelet.astype(np.uint8)
  medials=(medials).astype(np.uint8)
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
  propsTable=regionprops_table(labels,properties=('label','area','coords','perimeter','slice'))
  propsTable=pd.DataFrame(propsTable) 
  propsTable.sort_values(by='area',ascending=False,inplace=True)
  lesionLabel=propsTable.iloc[0]['label'] #lezyonun olduğu labeli buldum
  for i in props: #lezyonun olmadığı pikselleri 0 yapıyorum
    if(i.label!=lesionLabel):
      for j in i.coords:
        labels[j[0]][j[1]]=0
  # io.imshow(labels,cmap=plt.cm.gray)
  # propsTable.head()

  #UZUNLUK BULMA DENEMESİ

  import math

  coords = np.column_stack(np.where(labels ==1))
  stentLengthInPixel=0.0
  for i in range(len(coords)-1):
    euclideanDist = math.sqrt( ((coords[i][0]-coords[i+1][0])**2)+((coords[i][1]-coords[i+1][1])**2) )
    if(euclideanDist==1.0):
      stentLengthInPixel+=1.0
    elif(euclideanDist==math.sqrt(2)):
      stentLengthInPixel+=1.41421356
    else:
      print("cart",i)
      continue
  # return stentLengthInPixel
  catheterDiameterInMilimeter=0.2*10
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