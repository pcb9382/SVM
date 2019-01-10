"""
1.机器学习支持向量机（SVM）
2.使用简化版的SMO算法
3.使用线性核函数
姓名：pcb
时间：2018.12.26
"""
import numpy
from numpy import *
import matplotlib.pyplot as plt

#--------------SMO算法中的辅助函数-------------------------------------------------------
"""
加载数据
"""
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

"""
在某个区间范围内随机选择一个数
"""
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

"""
用于调整大于H或者小于L的alpha值
"""
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

"""
简化版SMO的伪代码：
    创建一个alpha向0量并将其初始化为0向量
    当迭代次数小于最大迭代次数时（外循环）
        对数据集中的每个数据向量（内循环）：
            如果该向量可以被优化：
               随机选择另外一个数据向量
               同时优化这两个向量
               如果这两个向量都不能被优化，退出内循环
        如果所有向量都没有被优化，增加迭代数目，继续下一次循环 
"""
#简化版的SMO算法（序列最小优化算法）
def smoSimple(dataMatIn,classLabel,C,toler,maxIter):
    """
    :param dataMatIn:   数据集
    :param classLabel:  类别标签
    :param C:           常数C
    :param toler:       容错率
    :param maxIter:     退出前的最大循环次数
    :return:
    """
    dataMatrix=mat(dataMatIn)                              #将列表输入转换成矩阵，简化数学处理操作
    labelMat=mat(classLabel).transpose()                   #将类别列表转置，得到列向量中的每一行都和数据矩阵中的行一一对应
    b=0;
    m,n=shape(dataMatrix)                                  #得到行和列
    alphas=mat(zeros((m,1)))                               #构建一个alpha列矩阵，矩阵中元素都初始化为0
    iter=0                                                 #该变量存储的是在没有任何alpha改变的情况下遍历数据集的次数
                                                           #当变量达到maxIter时，函数结束运行并退出
    while(iter<maxIter):
        alphaPairsChanged=0                                #用来记录alpha是否发生了变化
        for i in range(m):

            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b     #第i个样本的类别预测
            Ei=fXi-float(labelMat[i])                                                   #预测结果和真实结果的误差

            #检查是否alpha是否可以优化，如果可以优化则进入优化过程
            #在if语句中不管是正间隔还是负间隔都会被测试
            #同时也要检查alphas的值，以保证其不能等于0或C
            #由于后面alphas大于C或者小于0将会被调整为0或者C，如果一旦alpha等于这两个值的话就表明他们已经在"边界"上了
            #就不能在减小或者增大了，因此也就不能再对他们进行优化了
            #并且如果误差小于容错率也不用优化了
            if((labelMat[i]*Ei<-toler)and(alphas[i]<C))or((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                j=selectJrand(i,m)    #随机选择第二个alpha的值，在完整的SMO算法中选择第二个alphas时改进
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b #计算第二个alpha的类别
                Ej=fXj-float(labelMat[j])                                               #误差

                #为alphaIold和alphaJold分配内存空间，用于新值和旧值之间的比较
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()

                #用于修改alphas[j],保证其在0-C之间
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])

                #如果L=H，则不做任何改变
                if L==H:
                    print('L==H')
                    continue

                #eta是alpha[j]的最优修改量，如果是0，则需要退出当前迭代过程
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue

                #检车alpha[j]是否有轻微的变化，如果有则退出for循环
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print('j not moving enough')
                    continue

                #对i进行修改，修改量与j相同，但方向相反
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #设置常数项
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if (0<alphas[i])and(C>alphas[i]):
                    b=b1
                elif(0<alphas[j])and(C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                #如果执行到这都不执行continue语句，则说明成功改变了一对alpha,同时可以增加alphaPairsChanged的值
                alphaPairsChanged+=1

                print('iter:%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))

        #在for循环外，需要检测alpha值是否做了更新，如果有更新，则需要吧iter设置为0后继续运行程序
        if (alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print('iteration number:%d'%iter)

    return b,alphas

#---------------------------------------------------------------------------------------

#----------将得到的支持向量以及分类超平面画出来-------------------------------------------
def plotSupportVector(dataArr,labelArr,alphas,b):
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
    plt.savefig('简化版SMO支持向量和超平面示意图.png')
    plt.show()

#---------------------------------------------------------------------------------------



def main():
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataArr,labelArr,0.5,0.001,40)
    plotSupportVector(dataArr,labelArr,alphas.getA(),b)


if __name__=='__main__':
    main()
