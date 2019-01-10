"""
1.机器学习支持向量机SVM
2.使用完整的SMO算法进行加速
3.封装在类中
4.使用线性核函数
姓名：pcb
时间：2018.12.26
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

class optStruct:

    #初始化函数
    def __init__(self,dataMatIn,classLabel,C,toler):
        self.X=dataMatIn                            #m维特征向量
        self.labelMat=classLabel                    #m维标签向量
        self.C=C                                    #常数C
        self.tol=toler                              #容错率
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))          #建立并初始化alphas举证
        self.b=0
        self.eCache=mat(zeros((self.m,2)))          #第一列给出是否有效的标志位，第二列给出实际的E值

    #计算E值并返回结果
    def calcEk(self,k):
        fXk=float(multiply(self.alphas,self.labelMat).T*(self.X*self.X[k,:].T))+self.b #用于计算第k个样本的类别预测
        Ek=fXk-float(self.labelMat[k])                                                 #预测结果和真实结果的误差
        return Ek

    #在某个区间范围内随机选择一个数
    def selectJrand(self,i, m):
        j = i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    #用于调整大于H或者小于L的alpha值
    def clipAlpha(self,aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    #用于选择第二个alphas的值（内循环中的启发式方法）
    #选择合适的第二个alphas值以保证在每次优化中采用最大的步长
    def selectJ(self,i,Ei):
        maxK=-1;maxDeltaE=0;Ej=0
        self.eCache[i]=[1,Ei]                                        #将输入值在缓存在设置为有效
        validEcacheList=nonzero(self.eCache[:,0].A)[0]               #构建非零表，返回非零E值所对应的alphas值
        if (len(validEcacheList))>1:
            for k in validEcacheList:
                if k==i:
                    continue
                Ek=self.calcEk(k)
                deltaE=abs(Ei-Ek)
                if (deltaE>maxDeltaE):                             #在所有值上进行循环，并选择其中使得改变最大的那个值
                    maxK=k
                    maxDeltaE=deltaE;
                    Ej=Ek
            return maxK,Ej

        else:                                                      #如果这是一次循环的话就随机选择一个alphas值进行计算
            j=self.selectJrand(i,self.m)
            Ej=self.calcEk(j)
        return j,Ej

    #计算误差值并存入缓存中，再对alphas值进行优化时会用到这个值
    def updataEk(self,k):
        Ek=self.calcEk(k)
        self.eCache[k]=[1,Ek]

    #寻找决策边界
    def innerL(self,i):
        Ei=self.calcEk(i)

        if((self.labelMat[i]*Ei<-self.tol)and(self.alphas[i]<self.C))or\
                ((self.labelMat[i]*Ei>self.tol)and(self.alphas[i]>0)):

            j,Ej=self.selectJ(i,Ei)                      #第二个alphas选择中的启发方式
            alphaIold=self.alphas[i].copy()
            alphaJold=self.alphas[j].copy()

            if(self.labelMat[i]!=self.labelMat[j]):
                L=max(0,self.alphas[j]-self.alphas[i])
                H=min(self.C,self.C+self.alphas[j]-self.alphas[i])
            else:
                L=max(0,self.alphas[j]+self.alphas[i]-self.C)
                H=min(self.C,self.alphas[j]+self.alphas[i])

            if L==H:
                print('L==H')
                return 0

            eta=2.0*self.X[i,:]*self.X[j,:].T-self.X[i,:]*self.X[i,:].T-self.X[j,:]*self.X[j,:].T
            if eta>=0:
                print('eta>=0')
                return 0

            #更新误差缓存
            self.alphas[j]-=self.labelMat[j]*(Ei-Ej)/eta
            self.alphas[j]=self.clipAlpha(self.alphas[j],H,L)
            self.updataEk(j)

            if(abs(self.alphas[j]-alphaJold)<0.00001):
                print('j not moving enough')
                return 0

            self.alphas[i]+=self.labelMat[j]*self.labelMat[i]*(alphaJold-self.alphas[j])
            self.updataEk(i)

            #计算b1,b2的值
            b1=self.b-Ei-self.labelMat[i]*(self.alphas[i]-alphaIold)*self.X[i,:]*self.X[i,:].T-\
               self.labelMat[j]*(self.alphas[j]-alphaJold)*self.X[i,:]*self.X[j,:].T
            b2=self.b-Ej-self.labelMat[i]*(self.alphas[i]-alphaIold)*self.X[i,:]*self.X[j,:].T-\
                self.labelMat[j]*(self.alphas[j]-alphaJold)*self.X[j,:]*self.X[j,:].T

            if (0<self.alphas[i])and(self.C>self.alphas[i]):
                self.b=b1
            elif(0<self.alphas[j])and(self.C>self.alphas[j]):
                self.b=b2
            else:
                self.b=(b1+b2)/2.0
            a1, a2 = shape(mat(self.b))
            if (a1 != 1) or (a2 != 1):
                a = 111
            return 1

        else:
            return 0

    #完整的PlattSMO的外循环代码
    def smoP(self,dataMatIn,classLabels,C,tolar,maxIter,kTup=('lin',0)):

        iter=0
        entireSet=True
        alphaPairsChanged=0

        #当迭代次数超过最大指定值，或遍历整个集合都未对任意的alpha进行修改就退出循环
        #这里的一次迭代定义为一次循环过程，而不管该循环具体做了什么，如果优化过程中存在波动就停止
        while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
            alphaPairsChanged=0
            if entireSet:
                #遍历所有可能的alphas的值
                for i in range(self.m):
                    alphaPairsChanged+=self.innerL(i)       #通过调用选择第二个alphas,并在可能是对其进行优化
                    print('fullSet,iter:%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))
                iter+=1
            else:
                nonBoundIs=nonzero((self.alphas.A>0)*(self.alphas.A<C))[0]
                #遍历所有可能的非边界alphas值，也就是不在边界0或C上
                for i in nonBoundIs:
                    alphaPairsChanged+=self.innerL(i)
                    print('non-bound,iter:%d i:%d,pairs changed:%d'%(iter,i,alphaPairsChanged))
                iter+=1

            if entireSet:
                entireSet=False
            elif (alphaPairsChanged==0):
                entireSet=True
                print('iteration number :%d'%iter)
        return self.b,self.alphas

    #使用SVM进行分类
    def classificationSVM(self,dataMat):
        X=self.X;labelMat=self.labelMat
        m,n=shape(X)
        w=zeros((n,1))
        for i in range(m):
            w+=multiply(self.alphas[i]*labelMat[i],X[i,:].T)    #计算出w
        classficationResult=sum(dataMat*w.T)+self.b
        if classficationResult>0:
            classifyResult=1
        else:
            classifyResult=-1

        return classifyResult


    #----------将得到的支持向量以及分类超平面画出来-------------------------------------------
    def plotSupportVector(self,dataArr,labelArr,alphas,b):
        #s首先得到alphas向量中大于0小于C的，并根据符合要求的alpha找到支持向量
        alpha1=[]                        #存放符合要求的alpha值
        dataArr1=[];labelArr1=[]         #存放支持向量的坐标以及标签
        m,n=shape(alphas)
        xcord1 = [];ycord1 = []              #存放类别为1的数据坐标
        xcord2 = [];ycord2 = []              #存放类别为-1的数据坐标
        xcord3 = [];ycord3 = []              #存放类别为1的支持向量坐标
        xcord4 = [];ycord4 = []              #存放类别为-1的支持向量坐标
        for i in range(m):
            if alphas[i]>0:
                alpha1.extend(alphas[i])
                dataArr1.append(dataArr[i])
                labelArr1.append(labelArr[i])
                if labelArr[i]>0:
                    xcord3.append((dataArr[i])[0])
                    ycord3.append((dataArr[i])[1])
                else:
                    xcord4.append((dataArr[i])[0])
                    ycord4.append((dataArr[i])[1])

            if (labelArr[i]>0)and(alphas[i]<=0):
                xcord1.append((dataArr[i])[0])
                ycord1.append((dataArr[i])[1])
            if(labelArr[i]<0)and(alphas[i]<=0):
                xcord2.append((dataArr[i])[0])
                ycord2.append((dataArr[i])[1])

        #计算超平面

        m1=len(alpha1)
        #w =[];w2=[]
        w=mat(zeros((1,2)))
        dataArr2=mat(dataArr1)
        for i in range(m1):
            w+=dataArr2[i]*labelArr1[i]*alpha1[i]

        #得到超平面的系数
        a1=w[0,0]
        a2=w[0,1]

        #画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red')
        ax.scatter(xcord3,ycord3,s=50,c='red',marker='x',)
        ax.scatter(xcord2, ycord2, s=30, c='green', marker='s')
        ax.scatter(xcord4,ycord4,s=50,c='green',marker='x')
        ax.set_ylim(-8, 6)  # 设置纵轴范围，单独给图1设置y轴的范围
        x=arange(-2.0, 12.0, 1.0)
        y=(-b - a1 * x) / a2
        ax.plot(x,y.transpose())
        plt.title('Support Vector')
        plt.savefig('SVM支持向量和超平面示意图.png')
        plt.show()
    #---------------------------------------------------------------------------------------

#加载数据
def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat



def main():
    dataArr,lebalArr=loadDataSet('testSet.txt')
    oS=optStruct(mat(dataArr),mat(lebalArr).transpose(),0.6,0.001)           #定义一个类对象
    b,alphas=oS.smoP(dataArr,lebalArr,0.6,0.001,40)
    classifyResult=oS.classificationSVM(dataArr[20])
    print(classifyResult)
    oS.plotSupportVector(dataArr,lebalArr,alphas.getA(),b)


if __name__=='__main__':
    main()
