import math

class LorentzVector:
    def __init__(self):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.e = 0
        self.m = 0
        self.pt = 0

    def Px(self):
        return self.px

    def Py(self):
        return self.py

    def Pz(self):
        return self.pz

    def Pt(self):
        return self.pt

    def M(self):
        return self.m

    def E(self):
        return self.e
    
    def __add__(self,o):
        res = LorentzVector()
        res.SetPxPyPzE(self.px+o.px,self.py+o.py,self.pz+o.pz,self.e+o.e)
        return res
        

    def Print(self):
        print(" Px=",self.px," Py=",self.py, " Pz=",self.pz," Pt=",self.pt," E=",self.e," M=",self.m)

    def SetPxPyPzE(self,px,py,pz,e):
        self.px = px
        self.py = py
        self.pz = pz
        self.e = e
        self.pt = math.sqrt(px*px+py*py)
        m2 = e*e - (self.pt*self.pt + pz*pz) 
        if (m2>0):
            self.m = math.sqrt(m2)
        else:
            self.m = -math.sqrt(-m2)
