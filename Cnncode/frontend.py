import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # pip install pillow
from tkinter.filedialog import askopenfilename
import pytesseract
import cv2
import os
import numpy as np
import re


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\skula\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class FirstPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        load = Image.open("img1.jpg")
        photo = ImageTk.PhotoImage(load)
        label = tk.Label(self, image=photo)
        label.image=photo
        label.place(x=0,y=0)
        
        border = tk.LabelFrame(self, text='Login', bg='ivory', bd = 10, font=("Arial", 20))
        border.pack(fill="both", expand="yes", padx = 150, pady=150)
        
        L1 = tk.Label(border, text="Username", font=("Arial Bold", 15), bg='ivory')
        L1.place(x=50, y=20)
        T1 = tk.Entry(border, width = 30, bd = 5)
        T1.place(x=180, y=20)
        
        L2 = tk.Label(border, text="Password", font=("Arial Bold", 15), bg='ivory')
        L2.place(x=50, y=80)
        T2 = tk.Entry(border, width = 30, show='*', bd = 5)
        T2.place(x=180, y=80)
        
        def verify():
            try:
                with open("credential.txt", "r") as f:
                    info = f.readlines()
                    i  = 0
                    for e in info:
                        u, p =e.split(",")
                        if u.strip() == T1.get() and p.strip() == T2.get():
                            controller.show_frame(SecondPage)
                            i = 1
                            break
                    if i==0:
                        messagebox.showinfo("Error", "Please provide correct username and password!!")
            except:
                messagebox.showinfo("Error", "Please provide correct username and password!!")
         
        B1 = tk.Button(border, text="Submit", font=("Arial", 15), command=verify)
        B1.place(x=320, y=115)
        
        def register():
            window = tk.Tk()
            window.resizable(0,0)
            window.configure(bg="deep sky blue")
            window.title("Register")
            l1 = tk.Label(window, text="Username:", font=("Arial",15), bg="deep sky blue")
            l1.place(x=10, y=10)
            t1 = tk.Entry(window, width=30, bd=5)
            t1.place(x = 200, y=10)
            
            l2 = tk.Label(window, text="Password:", font=("Arial",15), bg="deep sky blue")
            l2.place(x=10, y=60)
            t2 = tk.Entry(window, width=30, show="*", bd=5)
            t2.place(x = 200, y=60)
            
            l3 = tk.Label(window, text="Confirm Password:", font=("Arial",15), bg="deep sky blue")
            l3.place(x=10, y=110)
            t3 = tk.Entry(window, width=30, show="*", bd=5)
            t3.place(x = 200, y=110)
            
            def validate_password(password):  
                if len(password) < 8:  
                    return False  
                if not re.search("[a-z]", password):  
                    return False  
                if not re.search("[A-Z]", password):  
                    return False  
                if not re.search("[0-9]", password):  
                    return False  
                return True
            
            def check():
                if t1.get()!="" or t2.get()!="" or t3.get()!="":                   
                    if validate_password(t2.get()) and t2.get()==t3.get():
                        with open("credential.txt", "a") as f:
                            f.write(t1.get()+","+t2.get()+"\n")
                            messagebox.showinfo("Welcome","You are registered successfully!!")
                    else:
                        messagebox.showinfo("Error","Your password is not Strong!!")
                else:
                    messagebox.showinfo("Error", "Please fill the complete field!!")
                    
            b1 = tk.Button(window, text="Sign in", font=("Arial",15), bg="#ffc22a", command=check)
            b1.place(x=170, y=150)
            
            window.geometry("470x220")
            window.mainloop()
            
        B2 = tk.Button(self, text="Register", bg = "dark orange", font=("Arial",15), command=register)
        B2.place(x=1000, y=20)

class SecondPage(tk.Frame):
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        
        self.configure(bg='white')
        
        #Label = tk.Label(self, text="Store some content related to your \n project or what your application made for. \n All the best!!", bg = "orange", font=("Arial Bold", 25))
        #Label.place(x=40, y=150)
        w2 = tk.Label(self, text="Signature Forgery Detection and Verification", bg="white"  ,fg="black"  ,width=50  ,height=1,font=('times', 30, 'italic bold underline'))
        w2.place(x=75,y=10)  
        
        '''
        l=Button(self,text="Build Training Model", command=self.buildModel, bg="red"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        l.place(x=200,y=200)
        '''
        k=tk.Button(self,text="Browse", command=self.showImgg, bg="black"  ,fg="white"  ,width=15  ,height=1,font=('times', 20, 'italic bold underline'))
        k.place(x=150,y=100)
        
        #t=tk.Button(self,text="ExtractOCR", command=self.OCR, bg="blue"  ,fg="white"  ,width=15  ,height=1,font=('times', 20, 'italic bold underline'))
        #t.place(x=380,y=100)
        
        #t=tk.Button(self,text="Classify for Signature", command=self.Classify1, bg="blue"  ,fg="white"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
        #t.place(x=500,y=100)
        
        Button = tk.Button(self, text="Next",command=lambda: controller.show_frame(ThirdPage),bg="black"  ,fg="white"  ,width=25  ,height=1,font=('times', 20, 'italic bold underline'))
        Button.place(x=850, y=500)
        
        t=tk.Button(self,text="Classify for Cheque Image", command=self.Classify, bg="black"  ,fg="white"  ,width=25  ,height=1,font=('times', 20, 'italic bold underline'))
        t.place(x=850,y=100)
                
        Button = tk.Button(self, text="Home",bg = "white", font=("Arial", 15), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=1100, y=50)
        
        Button = tk.Button(self, text="Back",bg = "white", font=("Arial", 15), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=100, y=50)
        
    def showImgg(self):
        self.load = askopenfilename(filetypes=[("Image File",'.png .jpg')])
        
        
        im = Image.open(self.load)
        
        im = im.resize((300, 150))
    
        render = ImageTk.PhotoImage(im)       
        
        # labels can be text or images
        img = tk.Label(self, image=render,width=300,height=150)
        img.image = render
        img.place(x=60, y=170)
    
    def OCR(self):
        please = 0
        sign = 0
        above = 0
        total_files = 0
        processed_files = 0
        print("Extract")
        img = cv2.imread(self.load)
        h, w, _ = img.shape  # assumes color image

        # Get verbose data including boxes, confidences, line, page numbers and text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([103, 79, 60])
        upper = np.array([129, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

        mask = 255 - mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        data = pytesseract.image_to_data(Image.open(self.load))
        pleaseCd = [0, 0, 0, 0]
        aboveCd = [0, 0, 0, 0]

        for d in data.splitlines():

            d = d.split("\t")
            print(d)

            if len(d) == 12:
                # # d[11] => text field of the image
                # # d[6] => left pointer of the image
                # # d[7] => right pointer of the image
                # # d[8] => width of the image
                # # d[9] => height of the image
                flag1 = 0
                # if(len(d[11]) == 12):
                #   print(d[11])
                if(len(d[11]) == 11):
                    s = d[11][:4]
                    temp = d[11]
                    # print(d[11])
                    if(s == "SYNB" or s == "SBIN" or s == "HDFC" or s == "CNRB" or s == "HDFC" or s == "PUNB" or s == "UTIB" or s == "ICIC"):
                        print("IFSC CODE : ", d[11])
                    if(s == "1C1C"):
                        str1 = temp
                        list1 = list(str1)
                        list1[0] = 'I'
                        list1[2] = 'I'
                        str1 = ''.join(list1)
                        print("IFSC CODE : ", str1)
                    # for x in range(5):
                    #     if(d[11][x] >= 'A' and d[11][x] <= 'Z'):
                    #         flag = 1
                    # if (flag == 1):
                    #     print(d[11])
                if d[11].lower() == "please":
                    pleaseCd[0] = int(d[6])
                    pleaseCd[1] = int(d[7])
                    pleaseCd[2] = int(d[8])
                    pleaseCd[3] = int(d[9])
                    please = please + 1
                if d[11].lower() == "sign":
                    sign = sign + 1
                if d[11].lower() == "above":
                    aboveCd[0] = int(d[6])
                    aboveCd[1] = int(d[7])
                    aboveCd[2] = int(d[8])
                    aboveCd[3] = int(d[9])
                    above = above + 1
        lengthSign = aboveCd[0] + aboveCd[3] - pleaseCd[0]
        scaleY = 2
        scaleXL = 2.5
        scaleXR = 0.5
        
        lengthSignCd = [0, 0, 0, 0]
        
        lengthSignCd[0] = int(pleaseCd[0] - lengthSign * 2.5)
        lengthSignCd[1] = int(pleaseCd[1] - lengthSign * 2)
        
        img = cv2.rectangle(
            img,
            (lengthSignCd[0], lengthSignCd[1]),
            (
                lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign),
                lengthSignCd[1] + int(scaleY * lengthSign),
            ),
            (255, 255, 255),
            2,
        )   
        cropImg = img[
            lengthSignCd[1]: lengthSignCd[1] + int(scaleY * lengthSign),
            lengthSignCd[0]: lengthSignCd[0]
            + int((scaleXL + scaleXR + 1) * lengthSign),
        ]
        if cropImg.size != 0:

            processed_files = processed_files + 1
            cv2.imwrite("imgcrop.jpg", cropImg)
        print("Done OCR")
    
    
    def linesweep(self):
        #images_dir = "../OCR/OCR_Results"
        # images_dir = "image"
        #input_path = os.path.join(
         #   os.path.dirname(os.path.abspath(__file__)), images_dir
        #)
        total_files = 0
        processed_files1 = 0

        #for filename in os.listdir(input_path):
        fileSize = os.stat("imgcrop.jpg").st_size
        total_files = total_files + 1
        if fileSize != 0:
                processed_files1 = processed_files1 + 1
                print("Processing imgcrop")
                img = Image.open("imgcrop.jpg")
                temp = np.array(img)

                grayscale = img.convert("L")
                xtra, thresh = cv2.threshold(
                    np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
                )

                # cv2.imshow('Binary Threshold', thresh)
                #
                # cv2.waitKey();

                # thresh = cv2.medianBlur(thresh, 2)

                rows = thresh.shape[0]
                cols = thresh.shape[1]

                flagx = 0
                indexStartX = 0
                indexEndX = 0

                for i in range(rows):
                    line = thresh[i, :]

                    if flagx == 0:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexStartX = i
                            flagx = 1
                            # print('start x: ', indexStartX, flagx)

                    elif flagx == 1:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexEndX = i
                            # print('end x: ', indexEndX)
                        elif indexStartX + 5 > indexEndX:
                            indexStartX = 0
                            flagx = 0
                            # print('elif x: ', indexStartX, flagx)
                        else:
                            break

                flagy = 0
                indexStartY = 0
                indexEndY = 0

                for j in range(cols):
                    line = thresh[indexStartX:indexEndX, j : j + 20]

                    if flagy == 0:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexStartY = j
                            flagy = 1
                            # print('start y: ', indexStartY, flagy)

                    elif flagy == 1:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexEndY = j
                            # print('end y: ', indexEndY)
                        elif indexStartY + 20 > indexEndY:
                            indexStartY = 0
                            flagy = 0
                            # print('elif y: ', indexStartY, flagy)
                        else:
                            break

                # print(indexStartX, indexEndX, indexStartY, indexEndY)
                cv2.line(
                    thresh,
                    (indexStartY, indexStartX),
                    (indexEndY, indexStartX),
                    (255, 0, 0),
                    1,
                )
                cv2.line(
                    thresh,
                    (indexStartY, indexEndX),
                    (indexEndY, indexEndX),
                    (255, 0, 0),
                    1,
                )

                cv2.line(
                    thresh,
                    (indexStartY, indexStartX),
                    (indexStartY, indexEndX),
                    (255, 0, 0),
                    1,
                )
                cv2.line(
                    thresh,
                    (indexEndY, indexStartX),
                    (indexEndY, indexEndX),
                    (255, 0, 0),
                    1,
                )
                temp_np = temp[
                    indexStartX : indexEndX + 1, indexStartY : indexEndY + 1
                ]

                # cv2.imshow('New cropped image', temp_np)
                #
                # cv2.waitKey();

                #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LineSweep_Results")
                #if not os.path.exists(path):
                #    os.makedirs(path)

                #s1 = "LineSweep_Result_crop_image"
                cv2.imwrite("LineSweep_Result_crop_image.jpg", temp_np)
                print("Done LINE_SWEEP")
    def Classify1(self):
        print("Classify the Test Image")
        #from tensorflow.keras.models import Model
        from tensorflow.keras.models import load_model
        #from keras.models import load_model
        model = load_model('signature_forgery_model3.h5')
        
        #Compiling the model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        #(model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']))
        #Making New Prediction
        import numpy as np
        from tensorflow.keras.preprocessing import image
        
        test_image = image.load_img(self.load,target_size = (64,64,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis = 0)
        result = model.predict(test_image)
        print(result)
        print(result[0])        
        
        if result[0] == 0.0:
            a = "forged.png"
            print("Forgery")
       
        else:
            a = "genuine.png"
            print("Original")
        
        im = Image.open(a)
        
        im = im.resize((300, 150))
    
        render = ImageTk.PhotoImage(im)       
        
        # labels can be text or images
        img = tk.Label(self, image=render,width=300,height=150)
        img.image = render        
        img.place(x=750,y=300)
    
    def Classify(self):
        print("Classify the Test Image")
        #from tensorflow.keras.models import Model
        from tensorflow.keras.models import load_model
        #from keras.models import load_model
        model = load_model('signature_forgery_model3.h5')
        
        #Compiling the model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        #(model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']))
        #Making New Prediction
        import numpy as np
        from tensorflow.keras.preprocessing import image
        
        please = 0
        sign = 0
        above = 0
        total_files = 0
        processed_files = 0
        print("Extract")
        img = cv2.imread(self.load)
        h, w, _ = img.shape  # assumes color image

        # Get verbose data including boxes, confidences, line, page numbers and text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([103, 79, 60])
        upper = np.array([129, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

        mask = 255 - mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        data = pytesseract.image_to_data(Image.open(self.load))
        pleaseCd = [0, 0, 0, 0]
        aboveCd = [0, 0, 0, 0]

        for d in data.splitlines():

            d = d.split("\t")
            print(d)

            if len(d) == 12:
                # # d[11] => text field of the image
                # # d[6] => left pointer of the image
                # # d[7] => right pointer of the image
                # # d[8] => width of the image
                # # d[9] => height of the image
                flag1 = 0
                # if(len(d[11]) == 12):
                #   print(d[11])
                if(len(d[11]) == 11):
                    s = d[11][:4]
                    temp = d[11]
                    # print(d[11])
                    if(s == "SYNB" or s == "SBIN" or s == "HDFC" or s == "CNRB" or s == "HDFC" or s == "PUNB" or s == "UTIB" or s == "ICIC"):
                        print("IFSC CODE : ", d[11])
                    if(s == "1C1C"):
                        str1 = temp
                        list1 = list(str1)
                        list1[0] = 'I'
                        list1[2] = 'I'
                        str1 = ''.join(list1)
                        print("IFSC CODE : ", str1)
                    # for x in range(5):
                    #     if(d[11][x] >= 'A' and d[11][x] <= 'Z'):
                    #         flag = 1
                    # if (flag == 1):
                    #     print(d[11])
                if d[11].lower() == "please":
                    pleaseCd[0] = int(d[6])
                    pleaseCd[1] = int(d[7])
                    pleaseCd[2] = int(d[8])
                    pleaseCd[3] = int(d[9])
                    please = please + 1
                if d[11].lower() == "sign":
                    sign = sign + 1
                if d[11].lower() == "above":
                    aboveCd[0] = int(d[6])
                    aboveCd[1] = int(d[7])
                    aboveCd[2] = int(d[8])
                    aboveCd[3] = int(d[9])
                    above = above + 1
        lengthSign = aboveCd[0] + aboveCd[3] - pleaseCd[0]
        scaleY = 2
        scaleXL = 2.5
        scaleXR = 0.5
        
        lengthSignCd = [0, 0, 0, 0]
        
        lengthSignCd[0] = int(pleaseCd[0] - lengthSign * 2.5)
        lengthSignCd[1] = int(pleaseCd[1] - lengthSign * 2)
        
        img = cv2.rectangle(
            img,
            (lengthSignCd[0], lengthSignCd[1]),
            (
                lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign),
                lengthSignCd[1] + int(scaleY * lengthSign),
            ),
            (255, 255, 255),
            2,
        )   
        cropImg = img[
            lengthSignCd[1]: lengthSignCd[1] + int(scaleY * lengthSign),
            lengthSignCd[0]: lengthSignCd[0]
            + int((scaleXL + scaleXR + 1) * lengthSign),
        ]
        if cropImg.size != 0:

            processed_files = processed_files + 1
            cv2.imwrite("imgcrop.jpg", cropImg)
        print("Done OCR")
        
        
        total_files = 0
        processed_files1 = 0

        #for filename in os.listdir(input_path):
        fileSize = os.stat("imgcrop.jpg").st_size
        total_files = total_files + 1
        if fileSize != 0:
                processed_files1 = processed_files1 + 1
                print("Processing imgcrop")
                img = Image.open("imgcrop.jpg")
                temp = np.array(img)

                grayscale = img.convert("L")
                xtra, thresh = cv2.threshold(
                    np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
                )

                # cv2.imshow('Binary Threshold', thresh)
                #
                # cv2.waitKey();

                # thresh = cv2.medianBlur(thresh, 2)

                rows = thresh.shape[0]
                cols = thresh.shape[1]

                flagx = 0
                indexStartX = 0
                indexEndX = 0

                for i in range(rows):
                    line = thresh[i, :]

                    if flagx == 0:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexStartX = i
                            flagx = 1
                            # print('start x: ', indexStartX, flagx)

                    elif flagx == 1:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexEndX = i
                            # print('end x: ', indexEndX)
                        elif indexStartX + 5 > indexEndX:
                            indexStartX = 0
                            flagx = 0
                            # print('elif x: ', indexStartX, flagx)
                        else:
                            break

                flagy = 0
                indexStartY = 0
                indexEndY = 0

                for j in range(cols):
                    line = thresh[indexStartX:indexEndX, j : j + 20]

                    if flagy == 0:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexStartY = j
                            flagy = 1
                            # print('start y: ', indexStartY, flagy)

                    elif flagy == 1:
                        ele = [255]
                        mask = np.isin(ele, line)

                        if True in mask:
                            indexEndY = j
                            # print('end y: ', indexEndY)
                        elif indexStartY + 20 > indexEndY:
                            indexStartY = 0
                            flagy = 0
                            # print('elif y: ', indexStartY, flagy)
                        else:
                            break

                # print(indexStartX, indexEndX, indexStartY, indexEndY)
                cv2.line(
                    thresh,
                    (indexStartY, indexStartX),
                    (indexEndY, indexStartX),
                    (255, 0, 0),
                    1,
                )
                cv2.line(
                    thresh,
                    (indexStartY, indexEndX),
                    (indexEndY, indexEndX),
                    (255, 0, 0),
                    1,
                )

                cv2.line(
                    thresh,
                    (indexStartY, indexStartX),
                    (indexStartY, indexEndX),
                    (255, 0, 0),
                    1,
                )
                cv2.line(
                    thresh,
                    (indexEndY, indexStartX),
                    (indexEndY, indexEndX),
                    (255, 0, 0),
                    1,
                )
                temp_np = temp[
                    indexStartX : indexEndX + 1, indexStartY : indexEndY + 1
                ]

                # cv2.imshow('New cropped image', temp_np)
                #
                # cv2.waitKey();

                #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LineSweep_Results")
                #if not os.path.exists(path):
                #    os.makedirs(path)

                #s1 = "LineSweep_Result_crop_image"
                cv2.imwrite("LineSweep_Result_crop_image.jpg", temp_np)
                print("Done LINE_SWEEP")
        
        test_image = image.load_img("LineSweep_Result_crop_image.jpg",target_size = (64,64,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis = 0)
        result = model.predict(test_image)
        print(result)
        print(result[0])        
        
        if result[0] == 0.0:
            a = "forged.png"
            print("Forgery")
       
        else:
            a = "genuine.png"
            print("Original")
            
        im1 = Image.open("LineSweep_Result_crop_image.jpg")
        
        im1 = im1.resize((300, 150))
    
        render1 = ImageTk.PhotoImage(im1)       
        
        # labels can be text or images
        img1 = tk.Label(self, image=render1,width=300,height=150)
        img1.image = render1        
        img1.place(x=200,y=400)
        
        im = Image.open(a)
        
        im = im.resize((300, 150))
    
        render = ImageTk.PhotoImage(im)       
        
        # labels can be text or images
        img = tk.Label(self, image=render,width=300,height=150)
        img.image = render        
        img.place(x=800,y=400)
        #controller.show_frame(ThirdPage)
        
        '''
        im = Image.open("graph.jpg")
        
        im = im.resize((300, 150))
    
        render = ImageTk.PhotoImage(im)       
        
        # labels can be text or images
        img = tk.Label(self, image=render,width=300,height=150)
        img.image = render        
        img.place(x=900,y=400)
        '''
 

class ThirdPage(tk.Frame):
    def __init__(self, parent, controller):        
        
        tk.Frame.__init__(self, parent)
        
        self.configure(bg='white')
        
        #Label = tk.Label(self, text="Store some content related to your \n project or what your application made for. \n All the best!!", bg = "orange", font=("Arial Bold", 25))
        #Label.place(x=40, y=150)
        w2 = tk.Label(self, text="Signature Forgery Detection and Verification", bg="white"  ,fg="black"  ,width=50  ,height=1,font=('times', 30, 'italic bold underline'))
        w2.place(x=75,y=10)  
        
        Button = tk.Button(self, text="Home",bg = "white", font=("Arial", 15), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=1100, y=50)
        
        Button = tk.Button(self, text="Back",bg = "white", font=("Arial", 15), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=100, y=50)
        
        
        k=tk.Button(self,text="View Accuracy and Loss Graph", command=self.graph, bg="black"  ,fg="white"  ,width=30  ,height=1,font=('times', 20, 'italic bold underline'))
        k.place(x=150,y=100)
        
        
    
    def graph(self):
        
        im = Image.open("graph_for_200Epochs.jpg")
        
        im = im.resize((500, 300))
    
        render = ImageTk.PhotoImage(im)       
        
        # labels can be text or images
        img = tk.Label(self, image=render,width=500,height=300)
        img.image = render        
        img.place(x=300,y=300)
                      
                
       
class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        #creating a window
        window = tk.Frame(self)
        window.pack()
        
        window.grid_rowconfigure(0, minsize = 800)
        window.grid_columnconfigure(0, minsize = 1300)
        
        self.frames = {}
        for F in (FirstPage, SecondPage, ThirdPage):
            frame = F(window, self)
            self.frames[F] = frame
            frame.grid(row = 0, column=0, sticky="nsew")
            
        self.show_frame(FirstPage)
        
    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()
        self.title("Application")
            
app = Application()
app.maxsize(1300,800)
app.mainloop()