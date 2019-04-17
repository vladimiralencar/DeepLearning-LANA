# Object Tracking com KCF

# Imports
import cv2
import sys
 
# Versão do OpenCV
minor_ver = 3
 
# Aplicação 
if __name__ == '__main__' :
 
    # Lista de Trackers
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

    # Definindo o tracker como KCF
    type=0
    tracker_type = tracker_types[type]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
 
    # Abre o vídeo para leitura
    video = cv2.VideoCapture("ball_green1.mp4")
 
    # Exit se o vídeo não pode ser aberto
    if not video.isOpened():
        print ("Não foi possível abrir o vídeo")
        sys.exit()
 
    # Leitura do primeiro frame
    ok, frame = video.read()
    if not ok:
        print ('Não é possível ler o arquivo')
        sys.exit()
     
    # Definir uma caixa inicial delimitadora
    bbox = (287, 23, 86, 320)
 
    # Descomente a linha abaixo para selecionar uma caixa delimitadora diferente
    bbox = cv2.selectROI(frame, False)
 
    # Inicialize o tracker com o primeiro frame e a caixa delimitadora
    ok = tracker.init(frame, bbox)
 
    while True:
        # Leitura de um novo frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Inicia o timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calcula os Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Desenha a bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking Falhou", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Mostra o tracker type no frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Mostra FPS no frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Mostra o resultado
        cv2.imshow("Tracking", frame)
 
        # Exit se a tecla ESC for pressionada
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        