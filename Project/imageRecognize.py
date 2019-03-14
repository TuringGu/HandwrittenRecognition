# -*- coding: utf-8 -*-
import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

# if r1 in r2, (r1 <= r2), return True
def inside(r1, r2):
    """judge weather r1 is in r2"""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if (x1 >= x2) and (y1 >= y2) and (x1 + w1 <= x2 + w2) and (y1 + h1 <= y2 + h2):
        return True
    else:
        return False


# Enlarging the rectangle range
def wrap_digit(rect):
    """
    Enlarging the rectangle range
    reshape it to square
    """
    x, y, w, h = rect
    padding = 5
    # padding = 20
    hcenter = x + w / 2
    vcenter = y + h / 2
    if (h > w):
        w = h
        x = hcenter - (w / 2)
    else:
        h = w
        y = vcenter - (h / 2)
    return (x - padding, y - padding, w + padding, h + padding)


# This funtion returns the pixel values.
# The imput is a png file location
def imageprepare(path):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    # file_name='./images/3.png'    # import your image dir
    # file_name = path
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(path).convert('L')

    # im.save("./image/sample.png")
    # plt.imshow(im)
    # plt.show()
    tv = list(im.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv]
    
    # print tva

    return tva



# recognize handwritten digits image, the input image is .png
def imageRecognize(filename):
    # png to jpg
    img_png = Image.open(filename)
    img_rgb = img_png.convert('RGB')
    img_rgb.save('test.jpg')
    
    font = cv2.FONT_HERSHEY_SIMPLEX # plot 
    
    # opencv read
    img_jpg = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)  # opencv read
    
    # color to gray
    img_gray = cv2.cvtColor(img_jpg, cv2.COLOR_BGR2GRAY)
    
    # GaussianBlur
    img_gray_gaussblur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    # binaryzation
    # ret, thbw = cv2.threshold(img_gray_gaussblur, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thbw = cv2.threshold(img_gray_gaussblur, 160, 255, cv2.THRESH_BINARY_INV)
    
    # Corroded image
    thbw = cv2.erode(thbw, np.ones((2, 2), np.uint8), iterations=2)
    
    # cv2.findContours()The function returns two values, one is the outline itself,
    # and the other is the attribute corresponding to each contour.
    image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    
    # traverse cntrs
    for i,c in enumerate(cntrs):
        r = x, y, w, h = cv2.boundingRect(c)    # use a smallest rectangle, wrap up it
        a = cv2.contourArea(c)  # The area of the contour
        
        # img.shape: show size
        # img.shape[0]: image width
        # img.shape[1]: image height
        # img.shape[2]: image channels
        # img.size: Display the total number of pixels
        # img.max(): max pixiv value
        # img.mean(): mean of pixivs value
        b = (img_jpg.shape[0] - 3) * (img_jpg.shape[1] - 3) # b is 3 pixivs reduced rectangle area
    
        is_inside = False   # initial
        
        # traverse rectangels
        for j,q in enumerate(rectangles):
            if inside(r, q):
                is_inside = True
                break
            if inside(q, r):
                rectangles.remove(q)
                pass
            
        # the image not in the rectangle
        if not is_inside:
            if not a == b:
                rectangles.append(r)    # append the smallest rectangle to the image
    
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # import MNIST dataset
    with tf.Graph().as_default() as g:
        x_ = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')  # input 28x28
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')# real value
    
        keep_prob = tf.placeholder("float")            
        y_conv = tf.nn.softmax(mnist_inference.inference(x_, keep_prob))
    
        saver = tf.train.Saver()
    
        # image segmentation and labeled
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("MNIST_model/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            accuracy = tf.argmax(y_conv, 1)
            
            i = 0
            for r in rectangles:
                i = i + 1
                x, y, w, h = wrap_digit(r)
                # TypeError: integer argument expected, got float
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                cv2.rectangle(img_jpg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = img_jpg[y:y + h, x:x + w]
            
                roi = cv2.resize(roi, (28,28))  # resize
                cv2.imwrite("test.png",roi)
                result = imageprepare("test.png")
                
                predint = accuracy.eval(feed_dict={x_: [result], keep_prob: 1.0}, session=sess)
                print('recognize result:',predint[0])
                cv2.putText(img_jpg, "%d" % predint, (x, y - 1), font, 2, (255, 0, 0), 4)

        
    # show the recognizition results
    cv2.imshow("thbw", thbw)
    cv2.imshow("contours", img_jpg)
    # cv2.imwrite("sample.jpg", img_jpg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return