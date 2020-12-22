import numpy as np
import matplotlib.pyplot as plt

"""
post processing vai
-----------------------------------------------------------------------------------
"""
with_crf = [0.84775298, 0.96464646, 0.9117823, 0.79647965, 0.84477612,
            0.73407407]
no_crf = [0.83734197, 0.92537313, 0.89388782, 0.79647965, 0.84736986,
          0.73]
assert len(with_crf) == len(no_crf)

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2  # width of bar
length = len(with_crf)
x = np.arange(length)

ax.bar(x, with_crf, width,label='with post-processing',color='#8a9d53')
ax.bar(x + width, no_crf, width, label='w/o post-processing',color='#ffa93a')

ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.15)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(['Roads', 'Water', 'Buildings', 'Cars', 'Trees', 'Grass'])
ax.legend()
fig.tight_layout()
plt.savefig('vai.png',dpi=300)
plt.show()


"""
post processing xiangliu
-----------------------------------------------------------------------------------
"""

# with_crf = [0.96969697, 0.95748792, 0.94684385, 0.91855392, 0.90764881, 0.87138264,
#             0.9751693, 0.98709239]
# no_crf = [0.86908359, 0.88719785, 0.88746752, 0.91171023, 0.89481481, 0.87911677,
#           0.90380313, 0.96454849]
# assert len(with_crf) == len(no_crf)
#
# # Set plot parameters
# fig, ax = plt.subplots()
# width = 0.25  # width of bar
# length = len(with_crf)
# x = np.arange(length)
#
# ax.bar(x, with_crf, width, label='with post-processing', color='#8a9d53')
# ax.bar(x + width, no_crf, width, label='w/o post-processing', color='#ffa93a')
#
# ax.set_ylabel('Accuracy')
# ax.set_ylim(0, 1.15)
# ax.set_xticks(x + width / 2)
# ax.set_xticklabels(['Fl. plants', 'Roads', 'Crops', 'Trees', 'Shrubs', 'Bare soil', 'Buildings', 'Water'])
# ax.legend(loc='upper right', framealpha=0.4)
# fig.tight_layout()
# plt.savefig('xiangliu.png', dpi=300)
# plt.show()


"""
multiscale strategy vai
-----------------------------------------------------------------------------------
"""
# monet1=[0.792,0.848,0.851,0.714,0.831,0.699]
# monet2=[0.817,0.903,0.894,0.797,0.839,0.733]
# monet3=[0.840,0.978,0.885,0.787,0.828,0.754]
# monet=[0.848,0.965,0.912,0.796,0.845,0.734]
#
# # Set plot parameters
# fig, ax = plt.subplots()
# width = 0.2  # width of bar
# length = len(monet)
# x = np.arange(length)
#
# ax.bar(x, monet1, width, label='MONet_1', color='#5d9bd6')
# ax.bar(x + width, monet2, width, label='MONet_2', color='#494cb5')
# ax.bar(x + 2*width, monet3, width, label='MONet_3', color='#ffca33')
# ax.bar(x + 3*width, monet, width, label='MONet', color='#ffad39')
#
# ax.set_ylabel('Accuracy')
# ax.set_ylim(0, 1.3)
# ax.set_xticks(x + 1.5*width)
# ax.set_xticklabels(['Roads', 'Water', 'Buildings', 'Cars', 'Trees', 'Grass'])
# ax.legend(loc='upper right', framealpha=0.4)
# fig.tight_layout()
# plt.savefig('multiscale_vai.png', dpi=300)
# plt.show()

"""
multiscale strategy xiangliu
-----------------------------------------------------------------------------------
"""
# monet1 = [0.676, 0.774, 0.813, 0.850, 0.830, 0.825, 0.821, 0.947]
# monet2 = [0.787, 0.842, 0.881, 0.891, 0.904, 0.869, 0.888, 0.962]
# monet3 = [0.850, 0.883, 0.916, 0.901, 0.924, 0.891, 0.927, 0.978]
# monet = [0.970, 0.957, 0.947, 0.919, 0.908, 0.871, 0.975, 0.987]
#
# # Set plot parameters
# fig, ax = plt.subplots()
# width = 0.2  # width of bar
# length = len(monet)
# x = np.arange(length)
#
# ax.bar(x, monet1, width, label='MONet_1', color='#5d9bd6')
# ax.bar(x + width, monet2, width, label='MONet_2', color='#494cb5')
# ax.bar(x + 2 * width, monet3, width, label='MONet_3', color='#ffca33')
# ax.bar(x + 3 * width, monet, width, label='MONet', color='#ffad39')
#
# ax.set_ylabel('Accuracy')
# ax.set_ylim(0, 1.3)
# ax.set_xticks(x + 1.5 * width)
# ax.set_xticklabels(['Fl. plants', 'Roads', 'Crops', 'Trees', 'Shrubs', 'Bare soil', 'Buildings', 'Water'])
# ax.legend(loc='upper right', framealpha=0.4)
# fig.tight_layout()
# plt.savefig('multiscale_xiangliu.png', dpi=300)
# plt.show()
