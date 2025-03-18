import cv2
import c

img = cv2.imread("/Users/hope/Desktop/uwut/deneme.png")
#a4 boyut ölçeklendirmesi yapma
scale = 3
wP = 210 *scale
hP = 297 *scale
img = cv2.resize(img,(0,0),None,0.5,0.5)
while True:
    #getcontours sayesinde elde ettiğimiz konturun alanı ve koordinatları contsta tutulur
    #imgcontours da ise konturlu resim tutulur
    imgContours , conts = c.getContours(img, minArea=50000,filter=4)

    if len(conts) !=0:
        #en büyük kontur seçiliyor
        biggest = conts[0][2]
        #en büyük kontura göre perspektif ayarlanıyor ve a4 boyutu atanıyor
        imgWarp = c.warpImg(img, biggest,wP,hP)
        #tekrardan kontur tespiti yapılıyor bu ise a4 ün üzerindeki cisimler için
        imgContours2, conts2 = c.getContours(imgWarp, minArea=2000, filter=4,cThr=[50,50],draw = False)
        if len(conts) != 0:
            for obj in conts2:
                #bu konturlar çiziliyor
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                #kontur noktaları sıralanıyor
                nPoints = c.reorder(obj[2])
                #iki nokta arasındaki mesafe hesaplanıyor 10 a bölünme sebebi ise cm cinsi lazım olması
                nW = round((c.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((c.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
                #şekle ok çiziliyor
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                #kenar uzunlukları cm olarak yazılıyor
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255, 0, 255), 2)
        cv2.imshow('A4', imgContours2)

        key =  cv2.waitKey(1)
        if key==27:break

  

    cv2.imshow('resim',img)
    cv2.waitKey(1)
