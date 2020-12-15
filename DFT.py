import numpy as np
import PIL.Image as Im
import copy

class dft():
    def __init__(self,img_n):
        #输入矩阵
        self.img_n=img_n
        M,N=img_n.shape
        img_F_n = self.__dft_line(img_n)
        # 傅立叶变换矩阵
        self.img_F_n = (self.__dft_line(img_F_n.T)/np.power(M*N,0.5)).T
        aaa=1
    #傅立叶变换
    def __dft_line(self,line):
        N = line.shape[1]
        y = np.arange(N).reshape((1, N))
        u = y.reshape((N, 1))
        W = np.exp(-2j * np.pi * u * y / N)
        return np.dot(line, W)

    #输入图像
    def img(self):
        img_n=np.abs(self.img_n)
        img_n[img_n<0]=0
        img_n[img_n>255]=255
        img_n=img_n.astype(np.uint8)
        return Im.fromarray(img_n)

    #傅立叶变换后图像
    def fimg(self,shift=True):
        img_n = np.abs(self.img_F_n)
        if(shift):
            trans_raw, trans_col=self.__shift(img_n)
            img_n=np.dot(trans_raw,img_n)
            img_n=np.dot(img_n,trans_col)
        img_n[img_n<0]=0
        img_n[img_n>255]=255
        img_n=img_n.astype(np.uint8)
        return Im.fromarray(img_n)

    def __shift(self,img_n):
        M=img_n.shape[0]
        N=img_n.shape[1]
        trans_raw=np.zeros((M,M))
        trans_col=np.zeros((N,N))
        for i in range(M):
            trans_raw[i][i]=1
        for i in range(N):
            trans_col[i][i] = 1

        c=M//2
        trans_raw_temp=copy.deepcopy(trans_raw[c:,:])
        trans_raw[M-c:, :]=trans_raw[:c,:]
        aaa = trans_raw[:M - c, :]
        trans_raw[:M-c, :]=trans_raw_temp

        c=N//2
        trans_col_temp=copy.deepcopy(trans_col[:,c:])
        trans_col[:, N-c:]=trans_col[:,:c]
        trans_col[:, :N-c]=trans_col_temp

        return (trans_raw,trans_col)



class idft():
    def __init__(self, img_n):
        # 傅立叶逆变换前矩阵
        self.img_n = img_n
        M, N = img_n.shape
        img_iF_n = self.__idft_line(img_n)
        #傅立叶逆变换后矩阵
        self.img_iF_n = (self.__idft_line(img_iF_n.T)/np.power(M*N,0.5)).T

        #傅立叶逆变换
    def __idft_line(self,line):
        N = line.shape[1]
        y = np.arange(N).reshape((1, N))
        u = y.reshape((N, 1))
        W = np.exp(2j * np.pi * u * y / N)
        return np.dot(line, W)

    #输入图像
    def img(self):
        img_n=np.abs(self.img_n)
        img_n[img_n<0]=0
        img_n[img_n>255]=255
        img_n=img_n.astype(np.uint8)
        return Im.fromarray(img_n)

    #傅立叶逆变换后图像
    def ifimg(self):
        img_n = np.abs(self.img_iF_n)
        img_n[img_n < 0] = 0
        img_n[img_n > 255] = 255
        img_n = img_n.astype(np.uint8)
        return Im.fromarray(img_n)

def test1(img):
    img_n = np.array(img)
    img_n = img_n[:, :, 0]
    F_n = dft(img_n)
    F_n.img().save("./输入.jpg")
    F_n.fimg(False).save("./输出.jpg")
    F_n.fimg().save("./输出_中心变换.jpg")
    iF_n=idft(F_n.img_F_n)
    iF_n.ifimg().save("./复原.jpg")

if __name__=="__main__":
    im=Im.open("./test.jpg")
    im=test1(im)
