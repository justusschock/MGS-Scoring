import os
import sys

import cv2
import av


# video = open('MVI_0245.MOV')
#
# stream = next(s for s in video.streams if s.type == 'video')
#
# for packet in video.demux(stream):
#     for frame in packet.decode():
#         # some other formats gray16be, bgr24, rgb24
#         img = frame.to_nd_array(format='bgr24')
#         cv2.imshow("Test", img)
#
#     if cv2.waitKey(1) == 27:
#         break
#
# cv2.destroyAllWindows()

container = av.open('MVI_0245.MOV')
for i, frame in enumerate(container.decode(video=0)):
    frame.to_image().save('sandbox/%04d.jpg' % i)
    if i > 5:
        break
