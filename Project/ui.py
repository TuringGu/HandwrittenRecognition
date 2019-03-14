#! /usr/bin/python
# -*- coding: utf-8 -*-

from Tkinter import *
# 导入tkinter模块的所有内容
import os
import shutil
import tkFileDialog
import tkFont
import mnist_inference
import imageRecognize
# import mnist_train
import mnist_eval
from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import string
import time
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50
# LEARNING_RATE_BASE = 0.8
# LEARNING_RATE_DECAY = 0.999
# REGULARIZATION_RATE = 0.0001
# TRAINING_STEPS = 30000
# MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"

# restore the model, and evaluate the accuracy of this model
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        # validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        
        keep_prob = tf.placeholder("float")
        y = mnist_inference.inference(x, keep_prob)
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
                print "test accuracy %g" % accuracy
                
                var.set(u"经过%s个训练步骤后,识别准确率为%g%%"%(global_step, accuracy*100))        
                root.update()
                return   global_step, accuracy*100
                #print(correct_prediction)
            else:
                print('No checkpoint file found')
                return

# model training
def train(mnist, training_steps):  
    
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    keep_prob = tf.placeholder("float")

    y_conv = tf.nn.softmax(mnist_inference.inference(x, keep_prob))
    global_step = tf.Variable(0, trainable=False)
    
    # CNN
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step = global_step)
    correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
    
    # clean the model saved path
    shutil.rmtree(MODEL_SAVE_PATH)  # delete file
    os.mkdir(MODEL_SAVE_PATH)   # create the same name file
    saver = tf.train.Saver()    # create the saver object
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(training_steps):
            steps = i
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i%100 == 0:
                # print train accuracy every 100 steps
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
                print "step %d, train accuracy %g" %(i, train_accuracy)
                
                # model saving per 1000 steps
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
               
            # feed_dict的作用是给使用placeholder创建出来的tensor赋值
            # keep_prob指定有多少个神经元是工作的
            _, loss= sess.run([train_step, cross_entropy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            
            # set step and loss to GUI
            if i%100 == 0:
                var.set(u"经过%d个训练步骤后，当前批次训练误差为%g%%。" % (i, loss))

            var1.set(u"当前完成率：%.2f%%" % ((i*100)/float(training_steps)))
            root.update()
            
            # plot the loss for current batch
            if (i+1)%100==0:
                global fig,ax,fig_y,fig_x
                fig_y.append(loss)
                fig_x.append(i)
                ax.set_title('Neural network training')
                ax.set_xlabel('Number_of_training')
                ax.set_ylabel('loss')
                ax.cla()
                ax.plot(fig_x,fig_y,label='loss', color='r', linewidth=1)
                ax.legend()
                plt.pause(0.000000000001)   # wait to plot
                
            root.update()   # update
            
        print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
    return steps, accuracy
    

# parameters set button callback function
def parameterSet():
    global training_steps   # define training_step
    global learning_rate    # define learning rate
    training_steps = int(training_step_entry.get())   # set training_step
    var_train_step_label.set("训练步数：%d" % training_steps)
    learning_rate = float(learning_rate_entry.get())  # set learning rate
    var_learning_rate_label.set("学习率：%f" % learning_rate)
    

# single testset picture prediction button callback function    
def openfile1():
    default_dir = "./test_picture"  # setting default open dir
    fname = tkFileDialog.askopenfilename(title=u"选择一张测试集图片",
                                         initialdir=(os.path.expanduser(default_dir)))
    global image,photo
    image = Image.open(fname)
    image=image.resize((256,256))   # image resize 
    photo = ImageTk.PhotoImage(image)
    global imgLabel
    imgLabel.configure(image = photo)
    root.update()   # update

    num=0   # num initial to zero
    
    # read the file name as testset image index
    fname = os.path.basename(fname) # capture the file name
    for ch in fname:
        if ch.isdigit():
            num=num*10
            num=num+int(ch)
    print(num)
    var.set("处理中，请稍候......")
    var1.set("当前处理时间计算中......")
    root.update()
    start_time=time.clock()
    
    # recognize
    num=mnist_eval.eval_main(num)  
    
    end_time=time.clock()
    var1.set(u"当前处理耗时%s秒"%(end_time-start_time))
    var.set("预测结果为%d"%num)
    
# single user-defined picture prediction button callback function
def openfile2():
    default_dir2 = "./user_defined_picture"  # setting default open dir
    fname2 = tkFileDialog.askopenfilename(title=u"选择一张自定义图片",
                                          initialdir=(os.path.expanduser(default_dir2)))
    global image,photo
    image = Image.open(fname2)
    image=image.resize((256,256))   # image resize 
    photo = ImageTk.PhotoImage(image)
    global imgLabel
    imgLabel.configure(image = photo)
    root.update()   # update
    
    # fname2 = os.path.basename(fname2) # capture the file name
    var.set("处理中，请稍候......")
    var1.set("当前处理时间计算中......")
    root.update()
    start_time=time.clock() 
    
    imageRecognize.imageRecognize(fname2)   # recognize
    
    # recognize
    # predint = sess.run(tf.argmax(y, 1), feed_dict={x: [result], keep_prob: 1.0})   # estimated value
    # num = predint[0]
    # print(predint)

    end_time=time.clock()
    var1.set(u"当前处理耗时%s秒"%(end_time-start_time))
    var.set("预测结果如图所示")
  

    
# model training button callback function
def callback1():         # 定义一个 改变文本的函数 .
    var.set("处理中，请稍后......")
    var1.set("当前处理时间计算中......")
    root.update()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    global fig,ax,fig_y,fig_x
    fig_y=[]
    fig_x=[]
    fig,ax=fig,ax=plt.subplots()
    start_time=time.clock()
    step,loss_value=train(mnist, training_steps)    # model training
    end_time=time.clock()
    var1.set(u"当前处理耗时%s秒"%(end_time-start_time))

# accuracy test button callback function
def callback2():         # 定义一个 改变文本的函数 .
    var.set("处理中，请稍后......")
    var1.set("当前处理时间计算中......")
    root.update()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    start_time=time.clock()
    step,loss_value=evaluate(mnist) # accuracy test
    end_time=time.clock()
    var1.set(u"当前处理耗时%s秒"%(end_time-start_time))
    
# window generate
def center_window(root, width, height):  
    screenwidth = root.winfo_screenwidth()  
    screenheight = root.winfo_screenheight()  
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width)/2, (screenheight - height)/2)
    root.geometry(size) 

# restart program
def restartProgram():
  python = sys.executable
  os.execl(python, python, * sys.argv)


# GUI
root = Tk()     # 初始化声明 . 
root.title("基于tensorflow的字符识别")
center_window(root, 700, 300)  
root.maxsize(1200, 800)  
root.minsize(600, 480)
frame1 = Frame(root)   # 在初始框架里面 声明两个模块 . 
frame2 = Frame(root)
frame3 = Frame(root)
frame4 = Frame(root)
global fig,ax,fig_y,fig_x
global counter
counter=1
# create a label object in frame1
global var
var = StringVar()           #声明可变 变量  . 
var.set("欢迎使用Tensorflow字符识别系统") # 设置变量 . 
textLabel = Label(frame1,           # 绑定到模块1
                  textvariable=var,  # textvariable 是文本变量的意思 .
                  font = '20',
                  justify=LEFT)    # 字体 位置
textLabel.pack(side=LEFT)   #  整体位置

# create a label object in frame2
global var1
var1 = StringVar()           #声明可变 变量  . 
var1.set("本程序使用Tensorflow进行数据处理，交互界面使用Tkinter") # 设置变量 . 
textLabel1 = Label(frame2,           # 绑定到模块1
                  textvariable=var1,  # textvariable 是文本变量的意思 .
                  font = '8',
                  justify=LEFT)    # 字体 位置 
textLabel1.pack(side=LEFT)   #  整体位置 

# 创建一个图像Label对象
# 用PhotoImage实例化一个图片对象（支持gif格式的图片）
global image,photo
image = Image.open("cover.jpg")
image=image.resize((256,256))
photo = ImageTk.PhotoImage(image)
global imgLabel
imgLabel = Label(frame4, image=photo)
imgLabel.pack(side=RIGHT)

# create training step label and entry in frame3
global var_train_step_label, var_learning_rate_label
var_train_step_label = StringVar()
var_learning_rate_label = StringVar()
var_train_step_label.set("训练步数：")
var_learning_rate_label.set("学习率：")
training_step_label = Label(frame3, textvariable=var_train_step_label, font = '15', justify = LEFT)
training_step_label.grid(row = 0, sticky = W)
training_step_entry = Entry(frame3)
training_step_entry.grid(row = 0, column = 1, sticky = E)
learning_rate_label = Label(frame3, textvariable=var_learning_rate_label, font = '15', justify = LEFT)
learning_rate_label.grid(row = 1, sticky = W)
learning_rate_entry = Entry(frame3)
learning_rate_entry.grid(row = 1, column = 1, sticky = E)
parameter_set_button = Button(frame3, text = "参数设置", font = '15', command = parameterSet)
parameter_set_button.grid(row = 2, column = 2, sticky = E)


# Add a button in frame4
# push the button, execute callback function
theButton1 = Button(frame4, text="模型训练", font = '15', command=callback1).pack()
theButton3 = Button(frame4, text="识别准确率检测", font = '15', command=callback2).pack()
theButton2 = Button(frame4, text="单个测试图像识别", font = '15', command=openfile1).pack()
theButton4 = Button(frame4, text="自定义图像识别", font = '15', command=openfile2).pack()
theButton5 = Button(frame4, text="重置", font = '15', command=restartProgram).pack(side='left')
theButton6 = Button(frame4, text="退出程序", font = '15', justify = LEFT, command=root.quit).pack(side='left')


# parameters adjust
frame1.pack(padx=10, pady=10)
frame2.pack(padx=5, pady=5)
frame3.pack(padx=10, pady=10)
frame4.pack(padx=10, pady=10)

mainloop()