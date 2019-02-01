import  os
import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt

path='image'
file=os.listdir(path)
num_predict=0


# plt.rcParams['figure.figsize']=(8.0,4.0)
# plt.rcParams['savefig.dpi']=400
# plt.rcParams['figure.dpi']=400
for cnt in range(len(file)):
    # if file[cnt][:4]=='seca':
    #     pass
    # else    :
    #     continue

    file_path = os.path.join(path, file[cnt])
    img0 = cv2.imread(file_path)
    print('\n'+file[cnt])


    img=cv2.resize(img0,(100,100))

    cv2.imshow('imgsd',img)
    imgcanny=cv2.Canny(img,100,200)
    cv2.imshow('imgcanny',imgcanny)

    canny_his = np.sum(imgcanny, axis=0)

    y_loc = 0
    flag = 0
    ycnt_0=0
    for i in range(canny_his.shape[0]-1):
        if canny_his[i]!=0:
            ycnt_0+=1
        if canny_his[i]!=0 and canny_his[i+1]==0:
            y_loc=i
        if  y_loc>0 and  canny_his[i]==0 and canny_his[i+1]!=0:
            flag=i
    print('y_loc:{0}'.format(y_loc))
    if ycnt_0>80:
        flag=0

    B=img[:,:,0]#.copy()
    G=img[:,:,1]#.copy()
    R=img[:,:,2]#.copy()

    Bhist = cv2.calcHist([B],[0],None,[256],[0,256]).flatten()
    Ghist = cv2.calcHist([G],[0],None,[256],[0,256]).flatten()
    Rhist = cv2.calcHist([R],[0],None,[256],[0,256]).flatten()

    N=50
    weight=np.ones(N)/N
    Bhist_conv =np.concatenate([np.zeros(1), np.convolve(weight, Bhist )[N - 1:-N + 1].astype(int),np.zeros(1)])
    Ghist_conv =np.concatenate([np.zeros(1),  np.convolve(weight, Ghist )[N - 1:-N + 1].astype(int),np.zeros(1)])
    Rhist_conv =np.concatenate([np.zeros(1),  np.convolve(weight, Rhist )[N - 1:-N + 1].astype(int),np.zeros(1)])

    #计算波峰。
    Bcnt=[0,]
    Bcnt_r=0
    for i in range(Bhist_conv.shape[0]-1):
        if Bhist_conv[i+1]>Bhist_conv[i]:
            Bcnt_r =i+1
        if Bhist_conv[i+1]<Bhist_conv[i] and Bcnt[-1]!=Bcnt_r:
            Bcnt.append(Bcnt_r)

    Gcnt=[0,]
    Gcnt_r=0
    for i in range(Ghist_conv.shape[0]-1):
        if Ghist_conv[i+1]>Ghist_conv[i]:
            Gcnt_r =i+1
        if Ghist_conv[i+1]<Ghist_conv[i] and Gcnt[-1]!=Gcnt_r:
            Gcnt.append(Gcnt_r)

    Rcnt=[0,]
    Rcnt_r=0
    for i in range(Rhist_conv.shape[0]-1):
        if Rhist_conv[i+1]>Rhist_conv[i]:
            Rcnt_r =i+1
        if Rhist_conv[i+1]<Rhist_conv[i] and Rcnt[-1]!=Rcnt_r:
            Rcnt.append(Rcnt_r)

    # print('Bcnt:{0}'.format(Bcnt[1:]))
    # print('Gcnt:{0}'.format(Gcnt[1:]))
    # print('Rcnt:{0}'.format(Rcnt[1:]))
    #####计算波峰end

    Bmax_diff=Bcnt[-1]-Bcnt[1]
    Gmax_diff=Gcnt[-1]-Gcnt[1]
    Rmax_diff=Rcnt[-1]-Rcnt[1]
    choose=np.argmax([Bmax_diff,Gmax_diff,Rmax_diff])
    img_thresh =R
    bigrate_pot=np.argmax(Rhist )
    thresh_loc=int((Rcnt[-1]+Rcnt[1])/2)
    if choose==0:
        img_thresh=B
        thresh_loc=int((Bcnt[-1]+Bcnt[1])/2)
        bigrate_pot=np.argmax(Bhist)
    elif choose==1:
        img_thresh=G
        thresh_loc=int((Gcnt[-1]+Gcnt[1])/2)
        bigrate_pot=np.argmax(Ghist )

    img_thresh=cv2.medianBlur(img_thresh,5)
    if bigrate_pot>127:
        _,img_thresh=cv2.threshold(img_thresh,thresh_loc+10,255,cv2.THRESH_BINARY_INV)
    else :
        _,img_thresh=cv2.threshold(img_thresh,thresh_loc-10,255,cv2.THRESH_BINARY)
    # img_thresh=cv2.resize(img_thresh,(20,20))
    # print('bigrate_pot:{0}'.format(bigrate_pot))
    # print('thresh_loc:{0}'.format(thresh_loc))
    # cv2.imshow('B', B)
    # cv2.imshow('G', G)
    # cv2.imshow('R', R)
    cv2.imshow('img_thresh', img_thresh)

    y_his = np.sum(img_thresh, axis=0)
    y_num = 1
    y_loc1 = 0
    flag1 = 0
    for i in range(canny_his.shape[0]-1):
        if canny_his[i]!=0 and canny_his[i+1]==0:
            flag1=i
        if  flag1>0 and  canny_his[i]==0 and canny_his[i+1]!=0:
            y_loc1=int((i+flag1)/2)
            y_num=2
            break

    if flag1>1 and flag==0:
        y_num=1

    num1=img_thresh
    num2=np.zeros([20,20])

    if y_num==2:
        num1=img_thresh[:,:y_loc1]
        num2 = img_thresh[:, y_loc1:]
        cv2.imshow('num2', num2)

    cv2.imshow('num1', num1)

    # img_thresh=cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=2)
    # img_thresh=img_thresh+imgcanny
    # cv2.imshow('img_threshwe', img_thresh)

    # x_his=np.sum(img_thresh,axis=1)
    # # print(x_his)
    #
    # x_loc=[0,255]
    # for i in x_his:
    #     if i==0:
    #        x_loc[0]+=1
    #     else :
    #         break
    #
    # for i in reversed(x_his):
    #     if i==0:
    #        x_loc[1]-=1
    #     else :
    #         break
    # print(x_loc)

    # y_his=np.sum(img_thresh,axis=0)/255
    # print(y_his)
    # ycnt=0
    # for i in y_his:
    #     if i==0:
    #         ycnt+=1
    # print('length:{0}'.format(100-ycnt))
    # num1 = np.arange(y_his.shape[0])
    # plt.figure()
    # plt.bar(num1, y_his, 0.5, color='green')
    # plt.savefig('hist_y_his\\y_' + file[cnt][:-4] + '.tif')
    # plt.show()

    if cv2.waitKey(0)==27:
        cv2.destroyAllWindows()



    # print(Rhist_conv)
    # Bhist_num1 = np.arange(Bhist_conv.shape[0])
    # Ghist_num1 = np.arange(Ghist_conv.shape[0])
    # Rhist_num1 = np.arange(Rhist_conv.shape[0])
    # plt.figure()
    # plt.subplot(131)
    # plt.bar(Bhist_num1, Bhist_conv, 0.5, color='green')
    # plt.subplot(132)
    # plt.bar(Ghist_num1, Ghist_conv, 0.5, color='green')
    # plt.subplot(133)
    # plt.bar(Rhist_num1, Rhist_conv, 0.5, color='green')
    # plt.savefig('hist\\hist_' + file[cnt][:-4] + '.tif')

    # for i in range(img.shape[1]):
    #     for ii in range(img.shape[0]):
    #         if B[i][ii] > 200 and G[i][ii] > 200 and R[i][ii] > 200:
    #             B[i][ii] = 0
    #             G[i][ii] = 0
    #             R[i][ii] = 0
    #         elif B[i][ii] < 100 and G[i][ii] < 100 and R[i][ii] < 100:
    #             B[i][ii] = 0
    #             G[i][ii] = 0
    #             R[i][ii] = 0


    #
    # _,imgB=cv2.threshold(R,127,255,cv2.THRESH_BINARY_INV)
    #
    # imgB=cv2.dilate(imgB,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))


    # cv2.imshow('imgB', imgB)

    # img1=imgB[:,:50]
    # img2=imgB[:,51:]
    #
    #
    # cv2.imshow('G', G)
    # cv2.imshow('R', R)








    # if high0>weight0:
    #     img2=cv2.copyMakeBorder(img1,0,0,int((high0-weight0)/2),int((high0-weight0)/2),cv2.BORDER_REPLICATE)
    # else:
    #     img2 = cv2.copyMakeBorder(img1, int((weight0 - high0) / 2), int((weight0 - high0) / 2),0, 0,
    #                               cv2.BORDER_REPLICATE)

    # img = cv2.resize(img2.copy(), (100, 100))
    # cv2.imshow('img', img)
    # imgblur=cv2.medianBlur(img,9)
    # cv2.imshow('imgblur', imgblur)
    #
    # imgf=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
    # _,imgf=cv2.threshold(imgf,150,255,cv2.THRESH_BINARY)
    # # imgCanny=cv2.Canny(imgblur,150,400)
    # cv2.imshow('imgCanny', imgf)

    #
    # Bhist_num1 = np.arange(256)
    # Ghist_num1 = np.arange(256)
    # Rhist_num1 = np.arange(256)
    # plt.figure()
    # plt.subplot(131)
    # plt.bar(Bhist_num1, Bhist, 0.5, color='green')
    # plt.subplot(132)
    # plt.bar(Ghist_num1, Ghist, 0.5, color='green')
    # plt.subplot(133)
    # plt.bar(Rhist_num1, Rhist, 0.5, color='green')
    # plt.savefig('hist\\hist_' + file[cnt][:-4]+'.tif')