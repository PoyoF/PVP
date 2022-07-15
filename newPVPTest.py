import cv2
import numpy as np
from colormath.color_objects import XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color
LuminanceRange=0.1 # 輝度変化範囲(-0.17～0.37)
#LuminanceRange=0.2 # 輝度変化範囲(-0.27～0.47)
#originalCWD=os.getcwd()
#TopFolder='C:\\imgs2\\' # 動画を置くところ
#変換関数
def xyY2sRGB(xyY):# xyY⇒RGB変換
    return convert_color(xyY, sRGBColor, target_illuminant='d50')
def RGB2BGR(a):# RGB⇒BGR変換
    return sRGBColor(a.rgb_b, a.rgb_g, a.rgb_r)
def PairPrint(p):# 色ペアデータ（辞書形式）のプリント
    print(p['name']+'Pair:\n\t',RGB2BGR(xyY2sRGB(p['p1'])),"\n\t",RGB2BGR(xyY2sRGB(p['p2'])))

#(おまけ)混同色線上の色から格子模様の色をセットする
#混同色線上の白以外の座標点
#protanope0 = (0.747, 0.253)
#deuteranope = (1.40, -0.40)
#tritanope = (0.171, 0.0)
#white = (1.0/3.0, 1.0/3.0)
#def getBothEndColors(p,radius):
#   global white
#   t = (radius / Distance(p, white))
#   return ((p - white) * t + white,-(p - white) * t + white)
#confusionLineLength = 0.1
#(p1,p2)=getBothEndColors(np.array(protanope0),confusionLineLength)
#
# Theory
#
#protanope (xp = 0.747, yp = 0.253), 
#deuteranope (xd = 1.40, yd = − 0.40), 
#tritanope(xt = 0.171, yt = 0)
#white(xw=1.0/3.0, yw=1.0/3.0)
#p=(protanope-white)*t+white  (±0.5)
#(p-white)^2=(protanope-white)^2*t^2=radius^2
#    t=±radius/abs(protanope-white)
#c.f. Humberto Moreira, Leticia Álvaro,Anna Melnikova and Julio Lillo(2017)"Colorimetry and Dichromatic Vision" Chapter 1: Colorimetry and Image Processing ,3-21 http://dx.doi.org/10.5772/intechopen.71563

# 色盲のタイプに合わせた、xy平面:中心(1/3,1/3)半径0.1の円上の点
#ProtanopePair:(0.42,0.32)--->(0.25,0.35)
#DeuteranopePair:(0.40,0.29)--->(0.26,0.38)
#TritanopePair:(0.30,0.26)--->(0.37,0.41)
# 色ペアデータ（辞書形式）、'name':文字列、'p1':xyYColor、'p2':xyYColor
ProtanopePair={'name':'Protanope','p1':xyYColor(0.42,0.32,1.0),'p2':xyYColor(0.25,0.35,1.0)}
DeuteranopePair={'name':'Deuteranope','p1':xyYColor(0.40,0.29,1.0),'p2':xyYColor(0.26,0.38,1.0)}
TritanopePair={'name':'Tritanope','p1':xyYColor(0.30,0.26,1.0),'p2':xyYColor(0.37,0.41,1.0)}
print("【 Color pairs 】最高輝度の座標")
PairPrint(ProtanopePair)
PairPrint(DeuteranopePair)
PairPrint(TritanopePair)

#輝度設定(xyY⇒xyY)
def setLuminance(xyYc,L): # 0.0<L<1.0
    xyYc.xyy_Y=L
    return xyYc
def forceRGB(c): # 取り合えずRGB値を求める。（マイナス輝度や255を超える輝度もあり得る）
    return tuple(map(int,np.array(list(c.get_value_tuple()),float)*255))
def RGBscale(c):# RGB値が0～255までの値に収まるようにする
    cs=forceRGB(c)
    return tuple(map(lambda x: 255 if(x>255) else x if(x>0) else 0,cs))
#　全ての色座標値が、表示可能な値となるような輝度基準値（中心輝度）を求める。
for c in range(255,0,-1):
    C=c/255.0
    Pp1RGBt=forceRGB(xyY2sRGB(setLuminance(ProtanopePair['p1'],C+LuminanceRange/2.0)))
    Pp2RGBt=forceRGB(xyY2sRGB(setLuminance(ProtanopePair['p2'],C+LuminanceRange/2.0)))
    Dp1RGBt=forceRGB(xyY2sRGB(setLuminance(DeuteranopePair['p1'],C+LuminanceRange/2.0)))
    Dp2RGBt=forceRGB(xyY2sRGB(setLuminance(DeuteranopePair['p2'],C+LuminanceRange/2.0)))
    Tp1RGBt=forceRGB(xyY2sRGB(setLuminance(TritanopePair['p1'],C+LuminanceRange/2.0)))
    Tp2RGBt=forceRGB(xyY2sRGB(setLuminance(TritanopePair['p2'],C+LuminanceRange/2.0)))
    if((max(Pp1RGBt)<255) and (max(Dp1RGBt)<255) and (max(Tp1RGBt)<255) and (max(Pp2RGBt)<255) and (max(Dp2RGBt)<255) and (max(Tp2RGBt)<255)
        and (min(Pp1RGBt)>0) and (min(Dp1RGBt)>0) and (min(Tp1RGBt)>0) and (min(Pp2RGBt)>0) and (min(Dp2RGBt)>0) and (min(Tp2RGBt)>0)):
        break
print("求められた中心輝度と輝度範囲:")
print("　CenterLuminance=",C,"LuminanceRange:",C-LuminanceRange/2.0,"～",C+LuminanceRange/2.0)
C=0.34 # 求められた中心輝度(0.345)
CenterLuminance=C # 計算された中心輝度
print("中心輝度の概算値と輝度範囲:")
print("　CenterLuminance=",C,"LuminanceRange:",C-LuminanceRange/2.0,"～",C+LuminanceRange/2.0)
Pp1RGBt=forceRGB(xyY2sRGB(setLuminance(ProtanopePair['p1'],C+LuminanceRange/2.0)))
Pp2RGBt=forceRGB(xyY2sRGB(setLuminance(ProtanopePair['p2'],C+LuminanceRange/2.0)))
Dp1RGBt=forceRGB(xyY2sRGB(setLuminance(DeuteranopePair['p1'],C+LuminanceRange/2.0)))
Dp2RGBt=forceRGB(xyY2sRGB(setLuminance(DeuteranopePair['p2'],C+LuminanceRange/2.0)))
Tp1RGBt=forceRGB(xyY2sRGB(setLuminance(TritanopePair['p1'],C+LuminanceRange/2.0)))
Tp2RGBt=forceRGB(xyY2sRGB(setLuminance(TritanopePair['p2'],C+LuminanceRange/2.0)))
print("Pp1RGBt:",tuple(Pp1RGBt))
print("Pp2RGBt:",tuple(Pp2RGBt))
print("Dp1RGBt:",tuple(Dp1RGBt))
print("Dp2RGBt:",tuple(Dp2RGBt))
print("Tp1RGBt:",tuple(Tp1RGBt))
print("Tp2RGBt:",tuple(Tp2RGBt))

def printColors(xyY):
    rgb=xyY2sRGB(xyY)
    RGBScale=RGBscale(rgb)
    print(" xyY=(%f,%f,%f)"%(xyY.xyy_x,xyY.xyy_y,xyY.xyy_Y),end="")
    print(" RGB=(%f,%f,%f)"%(rgb.rgb_r,rgb.rgb_g,rgb.rgb_b),end="")
    print(" Scaled=",RGBScale)
def printHLColors(xyY): # 最大輝度と最小輝度の色座標を表示する
    print("MaxLuminance: ",end="")
    printColors(setLuminance(xyY,CenterLuminance+LuminanceRange/2.0))
    print("MinLuminance: ",end="")
    printColors(setLuminance(xyY,CenterLuminance-LuminanceRange/2.0))
def printPair(p):#　等輝度化したい色ペアの表示関数
    print(p['name']+"-p1:")
    printHLColors(p['p1'])
    print(p['name']+"-p2:")
    printHLColors(p['p2'])

printPair(ProtanopePair)  #　等輝度化したい色ペアの表示
printPair(DeuteranopePair)#　等輝度化したい色ペアの表示
printPair(TritanopePair)  #　等輝度化したい色ペアの表示
####################################################
(W, H) = (500,20)         # 画像幅, 画像高さ
dW=5                      # 画像刻み幅
fps = 45.0                # [FPS]（Frame Per Second：１秒間に表示するFrame数）
stripeON=True
path=('S' if(stripeON) else 'N')+str(fps)+'c'+str(CenterLuminance)+'r'+str(LuminanceRange)
####################################################
def makeTestImages(p,path):
    global stripeON
    #pushedCWD=os.getcwd()
    #os.chdir(TopFolder)
    dataDir=p['name']
    filename = path+list(dataDir)[0]+'x='+str(p['p1'].xyy_x)+'y='+str(p['p1'].xyy_y)+"⇔"+'x='+str(p['p2'].xyy_x)+'y='+str(p['p2'].xyy_y)#ファイル名生成
    imgp1 = np.zeros((H,W,3), np.uint8)
    imgp2 = np.zeros((H,W,3), np.uint8)
    for x in range(W):
        xL=(CenterLuminance+LuminanceRange*((x/W)-0.5)) # 0～1.0の値
        xR=(CenterLuminance-LuminanceRange*((x/W)-0.5)) # 0～1.0の値
        p1=RGBscale(xyY2sRGB(setLuminance(p['p1'],xL)))
        p2=RGBscale(xyY2sRGB(setLuminance(p['p2'],xR)))
        for y in range(H):
            imgp1[y][x]=p1
            imgp2[y][x]=p2
    imgp1BGR = cv2.cvtColor(imgp1, cv2.COLOR_RGB2BGR) # BGRに変換
    imgp2BGR = cv2.cvtColor(imgp2, cv2.COLOR_RGB2BGR) # BGRに変換
    if(stripeON):
        for k in range(1,W//dW):        # 黒い縦線
            imgp1BGR = cv2.line(imgp1BGR,(k*dW,0),(k*dW,H-1),(0,0,0),1)
            imgp2BGR = cv2.line(imgp2BGR,(k*dW,0),(k*dW,H-1),(0,0,0),1)
    print(filename)
    #os.chdir(pushedCWD)
    return imgp1BGR,imgp2BGR

from PIL import Image
from logging import PlaceHolder
import streamlit as st
st.title("PVP Viewer")
import cv2
from PIL import Image
import numpy as np
# Streamlitの用意
PlaceHolderP = st.empty()
with PlaceHolderP.container():
    imageP=st.empty()
    posP = st.slider("Position", min_value=0, max_value=500, step=1, value=250,key='P')
PlaceHolderT = st.empty()
with PlaceHolderT.container():
    imageT=st.empty()
    posT = st.slider("Position", min_value=0, max_value=500, step=1, value=250,key='T')
#配列変換
def cv2pil(image):# OpenCV型 -> PIL型
    return Image.fromarray(image.copy())
def pil2cv(image):# PIL型 -> OpenCV型
    return cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
###########################################################################
#  各ペアについて視覚刺激用動画ファイルを生成する
###########################################################################
#imageAD,imageBD=makeTestImages(DeuteranopePair,path)# (緑色の色覚障害) 2型2色覚のヒトの混同色線
imageAT,imageBT=makeTestImages(ProtanopePair,path)  # (赤色の色覚異常) 1型2色覚のヒトの混同色線
imageAP,imageBP=makeTestImages(TritanopePair,path)  # (青色の色覚障害) 3型2色覚のヒトの混同色線

print("done")

frameAT=cv2pil(imageAT)
print(imageAT.shape)#フレーム次元表示
black=np.full_like(frameAT,0)
blk=pil2cv(black)
blackBar = blk[:,posT:posT+20]
h, w = imageAT.shape[:2]
count=0
def bFrame(img,pos):
    global h,v,blackBar
    M = np.array([[1, 0, pos-25], [0, 1, 0]], dtype=float)
    M2 = np.array([[1, 0, pos+5], [0, 1, 0]], dtype=float)
    tmp = cv2.warpAffine(blackBar, M, (w, h),img.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    return cv2pil(cv2.warpAffine(blackBar, M2, (w, h),tmp, borderMode=cv2.BORDER_TRANSPARENT))

def atFrame(pos,count):
    global AT,BT
    if(count%2):
        return AT[pos]
    else:
        return BT[pos]
def apFrame(pos,count):
    global AP,BP
    if(count%2):
        return AP[pos]
    else:
        return BP[pos]
AT=[]
BT=[]
AP=[]
BP=[]
for pos in range(500):
    AT.append(bFrame(imageAT,pos))
    BT.append(bFrame(imageBT,pos))
    AP.append(bFrame(imageAP,pos))
    BP.append(bFrame(imageBP,pos))
if 'count' not in st.session_state:
  st.session_state["count"] = 0
  
if (st.button('OK', key='G')):
    st.session_state["count"] +=1
    print("posT: ",posT,"    posP: ",posP)

while(True):
    imageT.image(atFrame(posT,count), caption='tritanope')  #use_column_width=True
    imageP.image(apFrame(posP,count), caption='protanope')  #use_column_width=True
    count+=1

cap.release()