import numpy as np
import cv2 as cv
import datetime
cap = cv.VideoCapture(0)
# history, dist2Threshold, detectShadows
# history, varThreshold, detectShadows

lista_cnts = list()
mask = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=35, detectShadows=True)
# mask = cv.createBackgroundSubtractorKNN(history=150, dist2Threshold=90, detectShadows=True)
if not cap.isOpened():
    print('No se pudo abrir la camara')
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print('No se pudo recibir el frame. Saliendo.')
        break
    # Our operations on the frame come here
    foto = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mascara = mask.apply(gray)

    ret2, thresh = cv.threshold(mascara, 12, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for cnt in contours:
            if cv.contourArea(cnt) > 6000:
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imwrite('captura/{}.jpg'.format(datetime.datetime.now().isoformat()), foto)
        print('Foto tomada')

    # Display the resulting frame
    cv.imshow('frame', foto)
    # cv.imshow('frame', gray)
    cv.imshow('mask', mascara)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
