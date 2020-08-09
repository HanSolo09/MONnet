from predict import *


gt_probs=[.6,.7,.8,.9]
sxy_bilaterals=[60,70,80]
srgb_bilaterals=[10,15,20]

max_oa=0.85
para_list=[]
for gt_prob in gt_probs:
    for sxy_bilateral in sxy_bilaterals:
        for srgb_bilateral in srgb_bilaterals:
            do_crf(exclude_labels=[3],gt_prob=gt_prob,sxy_bilateral=sxy_bilateral,srgb_bilateral=srgb_bilateral)
            oa=do_evaluation('./data/vaihingen_final/test_list.csv', ignore_zero=False)
            if oa > max_oa:
                print(str(gt_prob)+' , '+str(sxy_bilateral)+' , '+str(srgb_bilateral))
                para_list = []
                para_list.append(gt_prob)
                para_list.append(sxy_bilateral)
                para_list.append(srgb_bilateral)
                max_oa=oa

print(para_list)
print(max_oa)

