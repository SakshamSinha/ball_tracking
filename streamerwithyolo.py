import av
import numpy
import tellopy
from cv2 import cv2
import caffe

def videoFrameHandler(event, sender, data):
    with open('tmpvid', 'wb') as w:
        w.write(data)

def encode(frame, ovstream, output):
    try:
        pkt = ovstream.encode(frame)
    except Exception:
        return False
    if pkt is not None:
        try:
            output.mux(pkt)
        except Exception:
            print('mux failed: ' + str(pkt))
    return True

def main():
    # drone = tellopy.Tello()
    # drone.log.set_level(2)
    # drone.connect()
    # drone.start_video()
    # drone.subscribe(drone.EVENT_VIDEO_FRAME, videoFrameHandler)

    container = av.open('ball_tracking_example.mp4')
    # container = av.open(drone.get_video_stream())
    video_st = container.streams.video[0]
    output = av.open('archive.mp4', 'w')
    ovstream = output.add_stream('mpeg4', video_st.rate)
    ovstream.pix_fmt = 'yuv420p'
    ovstream.width = video_st.width
    ovstream.height = video_st.height

    net = caffe.Net('mobilenet-yolov3.prototxt','mobilenet_yolov3_lite_deploy.caffemodel')

    counter = 0
    for packet in container.demux((video_st,)):
        for frame in packet.decode():
            image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            print(type(image))
            cv2.imshow('frame', image)

            new_frame = av.VideoFrame(width=frame.width, height=frame.height, format=frame.format.name)
            for i in range(len(frame.planes)):
                new_frame.planes[i].update(frame.planes[i])
            encode(new_frame, ovstream, output)
            counter += 1
            print("Frames encoded:", counter)
        if counter > 500:
            output.close()
            break

if __name__ == '__main__':
    main()