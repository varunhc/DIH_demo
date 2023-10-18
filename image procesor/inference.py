# -*- coding: utf-8 -*-
### Env Setup
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings('ignore', category=DeprecationWarning)
#warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import numpy as np
import cv2 as cv
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt

from scipy import stats

DENOISED_MODEL_PATH = 'models/denoised/'
CLASSIF_MODEL_PATH = 'models/model_id_0/'

denoised_model = load_model(DENOISED_MODEL_PATH)
print('Denoising Model loaded.')
classif_model = load_model(CLASSIF_MODEL_PATH)
print('Classification Model loaded.')      

class_ids = {"10μm": 0, "20μm": 1, "MCF7": 2, "HepG2": 3, "RBC": 4, "WBC": 5}
# setup plot details
colors = ['royalblue', 'moccasin', 'darkorange', 'yellow', 'aquamarine','chartreuse']
class_names = ["10μm", "20μm", "MCF7", "HepG2", "RBC","WBC"]
color_rgb = [(65,105,225), (255,228,181), (255,140,0), (255,255,0), (127,255,212), (127,255,0) ]

"""Segment DIH Cells"""

def new_segm(imgpath, bins=256, thresh=180):
    img_raw = cv.imread(imgpath, 0) 
    #cv.namedWindow( 'orig' , cv.WINDOW_NORMAL )
    #cv.imshow( 'orig' , img_raw )

    planes={}
    for k in range(5, 8):
        plane = np.full((img_raw.shape[0], img_raw.shape[1]), 2 ** k, np.uint8)
        res = cv.bitwise_and(plane, img_raw)
        x = res * 255   
        v = np.median(x)
        if v>0:
            x = cv.bitwise_not(x)
            v = np.median(x)
            x[x == v] = 0            
        planes[str(k)] = x

    #sq_ker = np.ones((3,3),np.uint8)    
    ell_ker= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    #rect_ker = cv.getStructuringElement(cv.MORPH_RECT,(3,3))

    planes['5'] = cv.erode( planes['5'], ell_ker, iterations = 1)
    planes['5'] = cv.dilate(planes['5'], ell_ker, iterations = 2)
    #planes['5'] = cv.fastNlMeansDenoising(planes['5'], 7, 21)

    planes['765'] = planes['7'] + planes['6'] + planes['5']

    erosion1 = cv.erode(planes['765'], ell_ker, iterations = 1)
    erosion1 = cv.dilate(erosion1, ell_ker, iterations = 1)
    erosion1 = cv.erode(erosion1, ell_ker, iterations = 1)
    erosion1 = cv.dilate(erosion1, ell_ker, iterations = 1)

    #cv.namedWindow( 'ero'+str('765') , cv.WINDOW_NORMAL )
    #cv.imshow( 'ero'+str('765') , erosion1 )

    ############################################################

    #img = planes['765'].copy()
    img = erosion1.copy()
    r , c  = erosion1.shape
    
    # global thresholding
    _, th1 = cv.threshold(erosion1, thresh, 255, cv.THRESH_BINARY)    
    #cv.namedWindow( "global thresh", cv.WINDOW_NORMAL )
    #cv.imshow("global thresh",th1)

    #img = img_raw.copy()

    contours, _ = cv.findContours(th1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #cimg =  cv.cvtColor(th1, cv.COLOR_GRAY2RGB)
    #cv.drawContours(cimg, contours, -1, (0,255,0), 1)

    #img_clr = cv.cvtColor(img_raw, cv.COLOR_GRAY2RGB)

    c_areas, cx, cy, diams = [], [], [], []
    list_cnt=0

    for i in range(len(contours)):
        cnt = contours[i]
        if cnt.shape[0] > 5:
            M = cv.moments(cnt)
            if M['m00'] != 0:

                cx.append( int(M['m10']/M['m00']) )
                cy.append( int(M['m01']/M['m00']) )

                area = cv.contourArea(cnt)
                c_areas.append(area)
                equi_diameter = np.sqrt(4*area/np.pi)
                diams.append(equi_diameter)

                #if c_areas[list_cnt] > 30 and c_areas[list_cnt] < 50:
                #if diams[list_cnt]/2 < 20:
                #cv.circle(cimg, (int(cx[list_cnt]),int(cy[list_cnt])), int(diams[list_cnt]/2), (255,0,0), 1)
                #cv.circle(cimg, (int(cx[list_cnt]),int(cy[list_cnt])), 1, (0,0,255), 1)
                #cv.circle(img_clr, (int(cx[list_cnt]),int(cy[list_cnt])), int(diams[list_cnt]/2), (255,0,0), 1)
                #cv.circle(img_clr, (int(cx[list_cnt]),int(cy[list_cnt])), 1, (0,0,255), 1)
                #cv.rectangle(img_clr,(int(cx[list_cnt])-33 , int(cy[list_cnt])-33),(int(cx[list_cnt])+33 , int(cy[list_cnt])+33), (0,255,0), 3)            
                list_cnt+=1

    c_areas = np.array(c_areas)  
    cx = np.array(cx)      
    cy = np.array(cy)      
    diams = np.array(diams)   

    #cv.namedWindow( "contoured1", cv.WINDOW_NORMAL )
    #cv.imshow("contoured1",cimg)
    #cv.namedWindow( "contoured_img", cv.WINDOW_NORMAL )
    #cv.imshow("contoured_img",img_clr)

    #diam_mode = stats.mode(diams)
    #print("diam mode",diam_mode)
    #diam_mean = np.mean(diams)
    #print("diam mean", diam_mean)
    #diam_median = np.median(diams)
    #print("diam median", diam_median)
    #while True:
    #        key = cv.waitKey(1) & 0xFF
    #        if key == ord("e"): #delete recent-most point
    #            break 
    #cv.destroyAllWindows()
    
    return img_raw, cx, cy


def cropper(img, X, Y, csize=66):
    #I=img.copy()
    img_copy = img.copy()
    r,c = img.shape
    X,Y=np.int32(X), np.int32(Y)
    XY= list(zip(X,Y))    
    #I = cv.cvtColor(I, cv.COLOR_GRAY2RGB)
        
    cell_array=[]
    cell_id=[]
    
    for i in XY:
        x,y = i[0], i[1]
        #uncomment to mark all
        #stpt  = ( max(0,x-33), max(0,y-33) ) 
        #endpt = ( min(c,x+33), min(r,y+33) )
        
        #following codes removes cut out cells on the edges
        if x-33 >=0 and y-33 >=0 and x+33 <=c and y+33 <=r:
            stpt  = ( x-33, y-33)
            endpt = ( x+33, y+33) 
            
            indx = str(x)+"#"+str(y)
            cell_id.append(indx)            
            crop_cell = img_copy[ stpt[1]:endpt[1], stpt[0]:endpt[0] ].copy()                 
            cell_array.append(crop_cell)
            
            #cv.rectangle(I, stpt, endpt,  (255,255, 0), 1)            
            #cv.imshow("segmented_crop", crop_cell)
            #cv.waitKey(0)
            #cv.destroyAllWindows()            
            #break
            
    # cv.imshow("segmented", I)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.figure()
    # plt.figure(figsize=(15,20))
    # plt.imshow(I)
    # plt.savefig('final1.png',  bbox_inches='tight') #dpi=600,
    # plt.close()
    # plt.show()    
    return cell_id, cell_array


"""# Data Formatting"""
def test_eval(xtest): 
    xtest = np.array(xtest).reshape(-1,66,66,1).astype('float32')
    xtest=xtest/255.
    # print('xtest shape:', xtest.shape)
    # print(xtest.shape[0], 'test samples')
    return xtest

def testcell_resizer(xtest, new_size = 50, orig_size=66):
    pad = int( (orig_size-new_size)/2 )
    starti= 0 + pad
    endi=  66 - pad
    
    xts=[]
    for i in xtest:
        xts.append(i[starti:endi,starti:endi])    
    xts = np.asarray(xts).astype('float32')
    
    return xts


def code_runner(xtest, class_ids, cell_ids):
    seed=0          #seed for reproducibility    
    # csize= 50 #66
    # num_cls = 6 #10  
    
    #denoiser
    denoised_imgs = denoised_model.predict(xtest)
    #print(denoised_imgs.shape)

    #classifier
    y_pred = classif_model.predict_classes(denoised_imgs)
    #print(y_pred)            
    
    # only classifier
    # y_pred = classif_model.predict_classes(xtest)
    # print(y_pred)
    
    return list(zip(cell_ids, y_pred))
    #return list(zip(cell_ids, y_true, y_pred))

            
def disp_res(path, test_val, img_dir, imgname):    
    _, ypred = zip(*test_val)
       
    labels = class_names.copy()
    cell_counts = [ ypred.count(i) for i in range(len(class_names)) ]
    label_color = colors.copy()
    
    plt.figure()    #figsize=(10,10)
    bar_plot = plt.bar(labels, cell_counts, color= label_color)
    plt.ylabel('Cell Counts')
    plt.title('Cell Analysis')
    plt.xlabel('Cell types')

    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in bar_plot:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.legend(bar_plot, labels)    
    plt.tight_layout()
    plt.savefig(str('static/uploads/count_'+imgname), bbox_inches='tight', dpi=100)
    plt.close()
    #plt.show()    
    
    img=cv.imread(path,0)
    I=img.copy()
    I = cv.cvtColor(I, cv.COLOR_GRAY2RGB)
    
    for cell in test_val:
        cell_id, ypred = cell[0], cell[1]

        pos = cell_id.find('#')
        x,y = int(cell_id[:pos]), int(cell_id[pos+1:])
        #print(cell_id, x, y)

        stpt  = ( x-33, y-33)
        endpt = ( x+33, y+33)
        cv.rectangle(I, stpt, endpt,  color_rgb[ypred], 2)
        #cv.imshow("segmented_crop", crop_cell)
        #cv.waitKey(0)
        #cv.destroyAllWindows() 
        #break                   
    # cv.imshow("segmented", I)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    I = I[:,:,::-1]
    final_img_path = Path(img_dir) / str('result_'+imgname)
    cv.imwrite(str(final_img_path), I)    

    
def infer(imgname, img_dir):    
    print("Segmenting Cells...")
    
    imgpath = Path(img_dir) / imgname
    print(imgpath)
    #imgpath='dihsamples/test.png'

    img, X, Y = new_segm(str(imgpath), bins=256, thresh=180)
    cell_ids, cell_array = cropper(img, X, Y, csize=66)
    
    #cell_ids, cell_array = cell_segmenter(path=str(imgpath)) 
    print("Segmentation Complete...")
    
    cell_ids = np.asarray(cell_ids)
    cell_array = np.asarray(cell_array)
    # print(cell_ids.shape)
    # print(cell_array.shape)

    xtest = test_eval(cell_array)
    xtest = testcell_resizer(xtest)
    #print(xtest.shape)

    print("Analyzing Cells...")
    eval_results = code_runner(xtest, class_ids, cell_ids)
    
    disp_res( path = str(imgpath), test_val = eval_results, img_dir = img_dir, imgname = imgname)    