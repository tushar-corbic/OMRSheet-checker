

import cv2
import numpy as np
import utilis
import os
########################################################################
webCamFeed = False
pathImage = "2.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans=[]
print('Enter the number answers of all the',questions)
for i in range(questions):
    ans.append(int(input("Enter the answer of question")))
# images = utilis.load_images_from_folder('answerSheet')
folder = 'answerSheet'
imageList=[]
labels=[]
########################################################################

for filename in os.listdir(folder):
    p = folder + '/'+filename
    print(p)
    print(type(p))
    img = cv2.imread(p)
    print(os.path.join(folder, filename))
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY


    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
    rectCon = utilis.rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
    biggestPoints = utilis.getCornerPoints(rectCon[0])  # GET CORNER POINTS OF THE BIGGEST RECTANGLE
    gradePoints = utilis.getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

    if biggestPoints.size != 0 and gradePoints.size != 0:

        # BIGGEST RECTANGLE WARPING
        biggestPoints = utilis.reorder(biggestPoints)  # REORDER FOR WARPING
        cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET TRANSFORMATION MATRIX
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE

        # SECOND BIGGEST RECTANGLE WARPING
        cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # DRAW THE BIGGEST CONTOUR
        gradePoints = utilis.reorder(gradePoints)  # REORDER FOR WARPING
        ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
        matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE

        # APPLY THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

        boxes = utilis.splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES
        cv2.imshow("Split Test ", boxes[0][3])

        pixValues = np.zeros((questions, choices))
        for i in range(questions):
            for j in range(choices):
                pixValues[i][j] = cv2.countNonZero(boxes[i][j])
        # finding the marked option
        myIndex = []
        for i in range(questions):
            myIndex.append(np.argmax(pixValues[i]))



        # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
        grading = []
        for x in range(0, questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        # print("GRADING",grading)
        score = (sum(grading) / questions) * 100  # FINAL GRADE
        # print("SCORE",score)
        utilis.markScore(str(filename.partition('.')[0]),str(score))

        # DISPLAYING ANSWERS
        utilis.showAnswers(imgWarpColored, myIndex, grading, ans)  # DRAW DETECTED ANSWERS
        utilis.drawGrid(imgWarpColored)  # DRAW GRID
        imgRawDrawings = np.zeros_like(imgWarpColored)  # NEW BLANK IMAGE WITH WARP IMAGE SIZE
        utilis.showAnswers(imgRawDrawings, myIndex, grading, ans)  # DRAW ON NEW IMAGE
        invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
        imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

        # DISPLAY GRADE
        imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
        cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 85, 255), 3)  # ADD THE GRADE TO NEW IMAGE
        invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # INV IMAGE WARP

        # SHOW ANSWERS AND GRADE ON FINAL IMAGE
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
        cv2.imshow("Final Result", imgFinal)
        cv2.waitKey(10000)
        imageList.append(imgFinal)
        labels.append(filename.partition('.')[0])
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(filename.partition('.')[0]) + ".jpg", imgFinal)
        cv2.waitKey(300)








