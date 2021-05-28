from perception import *
from AdversarialAttack import LOAD
from Util import adversarial_detection,display_images

OriPref="./Result/AdversaryExample/Original_"
class Result:
    def __init__(self,filename,pref,color,label,marker,linestyle):
        self.Data = LOAD(filename)
        print(np.array(self.Data[0][0]).shape)
        self.pref = pref
        self.color = color
        self.label = label
        self.marker = marker
        self.linestyle=linestyle
        temp = [i[0][-1][1] for i in self.Data]
        temp1 = [i[1][-1][1] for i in self.Data]
        #print(temp1)
        count=0
        for item in temp1:
           if item<=16/255:#item!= float('inf'):
               count+=1
        print(count)
        self.rankindex = np.argsort(temp)
    def Plot(self,test,Number,Xrange,axe=None):
        x = range(Xrange)
        Y=[]
        for item in self.rankindex:
            X1 = [i[0] for i in self.Data[item][test]]
            Y1 = [i[1] for i in self.Data[item][test]]
            Y.append(np.interp(x, X1, Y1))
        Y = Y[0:Number]
        Y = np.mean(np.array(Y),0)
        print(Y[-1])
        if axe==None:
            plt.plot(x, Y,'-',color=self.color,label=self.label,marker=self.marker,linestyle=self.linestyle)
        else:
            axe.plot(x, Y,'-',color=self.color,label=self.label,marker=self.marker,linestyle=self.linestyle)
    def ASR(self,test,Number,Xrange,Thresh,axe=None):
        x = range(Xrange)
        Y=[]
        for item in self.rankindex:
            X1 = [i[0] for i in self.Data[item][test]]
            Y1 = [i[1] for i in self.Data[item][test]]
            Y.append(np.interp(x, X1, Y1))
        Y = np.asarray(Y[0:Number])
        temp = Y<Thresh
        Y[temp]=1
        Y[np.invert(temp)]=0
        Y = np.sum(Y,axis=0)
        Y = Y/Number
        if axe==None:
            plt.plot(x, Y,'-',color=self.color,label=self.label,marker=self.marker,linestyle=self.linestyle)
        else:
            axe.plot(x, Y,'-',color=self.color,label=self.label,marker=self.marker,linestyle=self.linestyle)
    def Hist(self,pref,Number,mode="median_smoothing"):
        Y = []
        for j in range(Number):
            i = self.rankindex[j]
            adv = np.load(pref + str(i) + ".npy")
            advdetect = adversarial_detection(adv, "median_smoothing")
            #if advdetect[0]:
            #    accadv += 1/rang
            Y.append(advdetect[1][0])
        plt.hist(Y, bins='auto',color=self.color,label=self.label)
    def Similiarity(self,pref,Number,mode="lpips"):
        Y = []
        for j in range(Number):
            i = self.rankindex[j]
            adv = np.load(pref + str(i) + ".npy")
            Ori = np.load(OriPref+str(i)+".npy")
            if j==0:
                display_images(adv)
                display_images(Ori)
            Y.append(special_distance(Ori,adv,mode))
        plt.hist(Y, bins='auto',color=self.color,label=self.label)
        print(np.mean(Y))