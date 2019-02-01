import  os
import numpy as np
from numpy.linalg import norm
import cv2
SZ=20
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 100, gamma = 0.3):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    #训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    #字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

model = SVM()
try:
    model.load("svm.dat")
except:
    print("error :no trained file!!")
    exit()

path='image'
file=os.listdir(path)
num_predict=0
for cnt in range(len(file)):
    if file[cnt][:4]=='seca':
        pass
    else    :
        continue

    file_path=os.path.join(path,file[cnt])
    img1=cv2.imread(file_path)
    print(file[cnt])
    img=cv2.resize(img1.copy(),(100,100))
    cv2.imshow('img',img)

    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Bhist = cv2.calcHist([B],[0],None,[256],[0,256]).flatten()
    Ghist = cv2.calcHist([G],[0],None,[256],[0,256]).flatten()
    Rhist = cv2.calcHist([R],[0],None,[256],[0,256]).flatten()

    Bhist_num1=np.sum(Bhist[:127])
    Ghist_num1=np.sum(Bhist[:127])
    Rhist_num1=np.sum(Ghist[:127])

    Bhist_num2 = np.sum(Bhist[128:])
    Ghist_num2 = np.sum(Bhist[128:])
    Rhist_num2 = np.sum(Ghist[128:])

    if Bhist_num1>Bhist_num2:
        Bhist_flag=1
    else:
        Bhist_flag=0
    if Ghist_num1>Ghist_num2:
        Ghist_flag=1
    else:
        Ghist_flag=0
    if Rhist_num1>Rhist_num2:
        Rhist_flag=1
    else:
        Rhist_flag=0

    print('{0};{1};{2}'.format(Bhist_flag,Ghist_flag,Rhist_flag))


    for i in range(100):
        for ii in range(100):
            if B[i][ii] > 200 and G[i][ii] > 200 and R[i][ii] > 200:
                B[i][ii] = 0
                G[i][ii] = 0
                R[i][ii] = 0
            elif B[i][ii] < 100 and G[i][ii] < 100 and R[i][ii] < 100:
                B[i][ii] = 0
                G[i][ii] = 0
                R[i][ii] = 0

    Bhist_light = cv2.calcHist([B], [0], None, [256], [0, 256]).flatten()[50:]
    Ghist_light = cv2.calcHist([G], [0], None, [256], [0, 256]).flatten()[50:]
    Rhist_light = cv2.calcHist([R], [0], None, [256], [0, 256]).flatten()[50:]
    #在BGR中寻找颜色分量最高的两个
    loc_array=np.array([np.max(Bhist_light),np.max(Ghist_light) ,np.max(Rhist_light)] )
    loc=np.argmax(loc_array)
    loc_array[loc]=0
    loc2 = np.argmax(loc_array)
    BGR_argmax_list=np.array([np.argmax(Bhist_light),np.argmax(Ghist_light),np.argmax(Rhist_light)])
    location = loc
    if BGR_argmax_list[loc]>BGR_argmax_list[loc2]:
        location=loc
    elif BGR_argmax_list[loc]<BGR_argmax_list[loc2]:
        location = loc2
    #确定选择哪个分量做主颜色
    if location==0:
        img_sd = B.copy()
        # print("B")
    elif location==1:
        img_sd = G.copy()
        # print("G")
    else:
        img_sd = R.copy()
        # print("G")
    #做颜色阈值分割
    _,img_sd=cv2.threshold(img_sd,127,255,cv2.THRESH_BINARY)
    img_sd=cv2.erode(img_sd,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    IMG_light = np.stack([B, G, R], axis=2)
    # cv2.imshow('IMG_light',IMG_light)
    #若白色分量较多，说明原始图片背景不为白色，是其他颜色，做反色处理
    white_count=np.sum(img_sd.flatten())/255
    # print('white_count:{0}'.format(white_count))
    if white_count>5000:
        _, img_sd = cv2.threshold(img_sd, 127, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('img_swd',img_sd)
        edge_img=cv2.Canny(img,150,400)
        img_sd+=edge_img
        img_sd=cv2.dilate(img_sd,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    #截顶和截底
    lightx_his = np.hstack([np.sum(img_sd, axis=1), np.zeros(1)]).astype(int)
    lightxloc = 0
    lightxloc_r = 0
    lightWeightx = 0
    for xcnt in range(len(lightx_his) - 1):
        if lightx_his[xcnt] == 0 and lightx_his[xcnt + 1] != 0:
            lightxloc_r = xcnt
        if lightx_his[xcnt] != 0 and lightx_his[xcnt + 1] == 0 and (xcnt - lightxloc_r) > lightWeightx:
            lightWeightx = xcnt - lightxloc_r
            lightxloc = lightxloc_r
        if lightWeightx > 5:
            break
    img_sd = img_sd[lightxloc:lightxloc + lightWeightx, :]
    ##分割数字
    y_his = np.hstack([np.sum(img_sd, axis=0), np.zeros(1)]).astype(int)/255-2
    # print(y_his)
    yloc = 0
    yloc_r = 0
    Weighty = 0
    for ycnt in range(len(y_his) - 1):
        if y_his[ycnt] <= 0 and y_his[ycnt + 1] > 0:
            yloc_r = ycnt
        if y_his[ycnt] > 0 and y_his[ycnt + 1] <= 0 and (ycnt - yloc_r) > Weighty:
            Weighty = ycnt - yloc_r
            yloc = yloc_r
        if Weighty > 5:
            break
    Weighty = Weighty + 1
    num1_area_sum=np.sum(y_his[yloc:yloc + Weighty])/Weighty
    num1 = img_sd[:, :yloc + Weighty]
    num2 = np.zeros(1)
    y_his[yloc:yloc + Weighty] = np.zeros(Weighty)
    num = 1
    if np.sum(y_his) != 0:
        yloc1 = 0
        yloc_r = 0
        Weighty2 = 0
        for ycnt in range(len(y_his) - 1):
            if y_his[ycnt] <= 0 and y_his[ycnt + 1] >= 0:
                yloc_r = ycnt
            if y_his[ycnt] >= 0 and y_his[ycnt + 1] <= 0 and (ycnt - yloc_r) > Weighty2:
                Weighty2 = ycnt - yloc_r
                yloc1 = yloc_r
            if (np.sum(y_his[yloc1:yloc1+Weighty2])/Weighty2) > 15.0:
                num = 2
        num2 = img_sd[:, yloc1:yloc1+Weighty2]
        num2 = cv2.resize(num2, (12, 24))
        num2 = cv2.copyMakeBorder(num2, 1, 1, 4, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if num1_area_sum<15:
            num1=num2
            num = 1
    #缩小形状以方便识别，并扩充边界
    num1 = cv2.resize(num1, (14, 24))
    num1 = cv2.copyMakeBorder(num1, 1, 1, 4, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    test = deskew(num1)
    pre1 = preprocess_hog([test])
    text1 = model.predict(pre1)
    print('text1:{0}'.format(text1[0].astype(int)))
    num_predict=text1[0].astype(int)
    # cv2.imshow('img_sd',img_sd)
    cv2.imshow('num1', num1)
    if num==2:
        cv2.imshow('num2',num2)
        test = deskew(num2)
        pre1 = preprocess_hog([test])
        text2 = model.predict(pre1)
        print('text2:{0}'.format(text2[0].astype(int)))
        num_predict = num_predict * 10 + text2[0].astype(int)
    print('num_predict:{0}'.format( num_predict ))
    if cv2.waitKey(0)==27:
        cv2.destroyAllWindows()