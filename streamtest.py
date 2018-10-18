import av
import io
import cv2
import tellopy
import time
import numpy



def videoFrameHandler(event, sender, data):
    with open('tmpvid', 'wb') as w:
        w.write(data)


def main():
    videostream = io.BytesIO()
    videostream.seek(0)

    drone = tellopy.Tello()
    drone.log.set_level(2)
    drone.connect()
    drone.start_video()
    drone.subscribe(drone.EVENT_VIDEO_FRAME, videoFrameHandler)

    #
    in_container = av.open(drone.get_video_stream())
    out_container = av.open('tmp.mp4', mode='w')
    # in tello.py it is set to 0x20
    out_stream = out_container.add_stream('mpeg4', rate=32)
    while True:
        for packet in in_container.demux((in_container.streams.video[0],)):
            for frame in packet.decode():
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                print("image", type(image))
                cv2.imshow('frame',image)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
        print(len(videostream))

if __name__ == '__main__':
    main()