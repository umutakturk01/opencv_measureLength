import cv2
import numpy as np

def getContours(img,cThr=[100,100] ,minArea=50000, filter=0, draw = False):
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#bgr formattaki matrixi gray formatına çevirir
    blur = cv2.GaussianBlur(gray,(5,5),1)#gürültüyü azaltmak için yapılır
    canny = cv2.Canny(blur,cThr[0],cThr[1])#köşeleri bulmak için yapılır
    kernel = np.ones((5,5)) #dilate ve erode fonksiyonunda kullanmak için kernel tanımlıyoruz
    dial = cv2.dilate(canny,kernel,iterations=3 ) #ince kenarları kalınlaştırır için kullanılır
    thre= cv2.erode(dial, kernel,iterations=2) # dilatede kalınlaşan kenarları çok fazla kalınlaşması için inceltilir
    
    # findcontours konturları bulur, retr_external fonskiyonu sadece dış konturları alır,
    # chain approx simple konturları sıkılaştırır daha az yer kaplamasına yarar
    # contours konturları içerir, hierchy hiyerarşik bilgiler içerir
    contours, hierchy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #seçilen konturler burada diziye kaydedilir
    finalCountours=[]
    
    for i in contours: #konturlar burada incelenir

        #konturun alanları alınır
        area = cv2.contourArea(i)

        #konturun alanı min alandan büyük mü diye kontrol edilir
        if area > minArea:
            
            peri = cv2.arcLength(i,True)#konturun çevresi alınır, true konturun kapalı olmasını kontrol eder
            #köşe sayısı azaltılarak daha düzgün çokgen elde edilir,
            #0.0.2 peri parametresi doğruluk toleransını belirler
            approx = cv2.approxPolyDP(i,0.02*peri,True) 
            #çokgenin etrafına yaklaşık bir dikdörtgen çizer
            bbox = cv2.boundingRect(approx)
            
            if filter > 0: # geometrik şekil olup olmadığını kontrol edilir
                if len(approx)==filter:
                    finalCountours.append((len(approx), area, approx, bbox, i)) #kontur listesine eklenir
                   

            else:#şekilin türüne bakmaz 
                finalCountours.append([len(approx), area, approx, bbox, i]) #kontur listesine eklenir

    #kontur sıralaması yapılır x[1]e göre azalan sıra ile yapılır
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)

    #konturları orijinal görüntüye çizer
    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3) #con[4] orijinal kontur bilgisi

    return img, finalCountours

#mypointsi belirli düzene göre sıralamak için kullanılır
def reorder(myPoints):
    
    myPointsNew =np.zeros_like(myPoints, dtype=np.int32) #mypoint ile aynı ama 0 lardan oluşan dizi oluşturur
    myPoints = myPoints.reshape((4,2)) # mypointsteki diziyi 4 nokta şeklinde [x,y] şeklinde düzenler
    add = myPoints.sum(1)  #x ve y noktalarını toplar sol üsten sağ alta doğru sıralamada yardımcı olsun diye
    myPointsNew[0]=myPoints[np.argmin(add)] #bulunan en küçük değer sol üst noktasına atanır
    myPointsNew[3]= myPoints[np.argmax(add)] #bulunan en büyük değer sağ alt noktasına atanır
    diff = np.diff(myPoints,axis=1) # x ve y nin arasındaki fark hesaplanır sağ üst ve sol alt noktasını tespit etmek amaçlı
    myPointsNew[1]= myPoints[np.argmin(diff)] # en küçük değer sağ üst noktasna atanır
    myPointsNew[2]= myPoints[np.argmax(diff)] # en büyük değer sol alt noktasına atanır
    return myPointsNew #mypointsnew dizisi döndürülür

#perspektif dönüşümü uygular
def warpImg(img, points,w,h,pad=20): # görüntü, dikdörtgenin koordinat noktaları, genişliği, yüksekliği, görüntünün etrafından kesilicek alan
    #köşe noktaları yeniden sıralanır sebebi ise perspektif dünüşümün doğru çalışması için sıralama önemli
    points = reorder(points)
    #float32 dönüşümünün sebebi: getPerspectiveTransform fonksiyonu bu tür girdi istediği için
    pts1 = np.float32(points)
    #perspektif dönüşümü sonrası elde edilen görselin koşe noktalarını belirler
    pts2 = np.float32([ [0,0], #sol üst köşe
                        [w,0], #sağ üst köşe
                        [0,h], #sol alt köşe
                        [w,h]]) #sağ alt köşe
    #iki nokta arasındaki perspektif dönüşüm matrixi hesaplanır 3x3 boyutunda matrix elde edilir
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    #img görseli matrixin boyutuna göre boyutlandırılır
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    #kenardaki boşluklar kesilir, kenardaki boşluklar kesilir
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    #iki nokta arasında öklid kuralı uygulanır mesafe ölçülür
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5



        

