import av
from PyQt5 import QtCore

from PyQt5.QtCore import QObject

AV_TIME_BASE = 1000000

def pts_to_frame(pts, time_base, frame_rate, start_time):
    return int(pts * time_base * frame_rate) - int(start_time * time_base * frame_rate)

def get_frame_rate(stream):

    if stream.average_rate.denominator and stream.average_rate.numerator:
        return float(stream.average_rate)
    if stream.time_base.denominator and stream.time_base.numerator:
        return 1.0 / float(stream.time_base)
    else:
        raise ValueError("Unable to determine FPS")

def get_frame_count(f, stream):

    if stream.frames:
        return stream.frames
    elif stream.duration:
        return pts_to_frame(stream.duration, float(stream.time_base), get_frame_rate(stream), 0)
    elif f.duration:
        return pts_to_frame(f.duration, 1 / float(AV_TIME_BASE), get_frame_rate(stream), 0)

    else:
        raise ValueError("Unable to determine number for frames")

class FrameGrabber(QObject):

    frame_ready = QtCore.pyqtSignal(object, object)
    update_frame_range = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.file = None
        self.stream = None
        self.frame = None
        self.active_frame = None
        self.start_time = 0
        self.pts_seen = False
        self.nb_frames = None

        self.rate = None
        self.time_base = None

    def next_frame(self):

        frame_index = None

        rate = self.rate
        time_base = self.time_base

        self.pts_seen = False

        for packet in self.file.demux(self.stream):
            #print "    pkt", packet.pts, packet.dts, packet
            if packet.pts:
                self.pts_seen = True

            for frame in packet.decode():

                if frame_index is None:

                    if self.pts_seen:
                        pts = frame.pts
                    else:
                        pts = frame.dts

                    if not pts is None:
                        frame_index = pts_to_frame(pts, time_base, rate, self.start_time)

                elif not frame_index is None:
                    frame_index += 1

                yield frame_index, frame


    @QtCore.pyqtSlot(object)
    def request_frame(self, target_frame):

        frame = self.get_frame(target_frame)
        if not frame:
            return

        img = frame.to_image().rotate(-90, expand=True)

        self.frame_ready.emit(img, target_frame)

    def request_next_frame(self, target_frame):
        index, frame = next(self.next_frame())

        if target_frame != index:
            raise ValueError("Indicides do not match. %i != %i" % (target_frame, index))

        if not frame:
            return

        img = frame.to_image().rotate(-90, expand=True)

        self.frame_ready.emit(img, index)

    def get_frame(self, target_frame):

        if target_frame != self.active_frame:
            return
        print('seeking to', target_frame)

        seek_frame = target_frame

        rate = self.rate
        time_base = self.time_base

        frame = None
        reseek = 250

        original_target_frame_pts = None

        while reseek >= 0:

            # convert seek_frame to pts
            target_sec = seek_frame * 1 / rate
            target_pts = int(target_sec / time_base) + self.start_time

            if original_target_frame_pts is None:
                original_target_frame_pts = target_pts

            self.stream.seek(int(target_pts))

            frame_index = None

            frame_cache = []

            for i, (frame_index, frame) in enumerate(self.next_frame()):

                # optimization if the time slider has changed, the requested frame no longer valid
                if target_frame != self.active_frame:
                    return

                print("   ", i, "at frame", frame_index, "at ts:", frame.pts, frame.dts, "target:", target_pts, 'orig', original_target_frame_pts)

                if frame_index is None:
                    pass

                elif frame_index >= target_frame:
                    break

                frame_cache.append(frame)

            # Check if we over seeked, if we over seekd we need to seek to a earlier time
            # but still looking for the target frame
            if frame_index != target_frame:

                if frame_index is None:
                    over_seek = '?'
                else:
                    over_seek = frame_index - target_frame
                    if frame_index > target_frame:

                        print(over_seek, frame_cache)
                        if over_seek <= len(frame_cache):
                            print("over seeked by %i, using cache" % over_seek)
                            frame = frame_cache[-over_seek]
                            break


                seek_frame -= 1
                reseek -= 1
                print("over seeked by %s, backtracking.. seeking: %i target: %i retry: %i" % (str(over_seek), seek_frame, target_frame, reseek))

            else:
                break

        if reseek < 0:
            raise ValueError("seeking failed %i" % frame_index)

        # frame at this point should be the correct frame

        if frame:

            return frame

        else:
            raise ValueError("seeking failed %i" % target_frame)

    def get_frame_count(self):

        frame_count = None

        if self.stream.frames:
            frame_count = self.stream.frames
        elif self.stream.duration:
            frame_count = pts_to_frame(self.stream.duration, float(self.stream.time_base), get_frame_rate(self.stream), 0)
        elif self.file.duration:
            frame_count = pts_to_frame(self.file.duration, 1 / float(AV_TIME_BASE), get_frame_rate(self.stream), 0)
        else:
            raise ValueError("Unable to determine number for frames")

        seek_frame = frame_count

        retry = 100

        while retry:
            target_sec = seek_frame * 1 / self.rate
            target_pts = int(target_sec / self.time_base) + self.start_time

            self.stream.seek(int(target_pts))

            frame_index = None

            for frame_index, frame in self.next_frame():
                print(frame_index, frame)
                continue

            if not frame_index is None:
                break
            else:
                seek_frame -= 1
                retry -= 1


        print("frame count seeked", frame_index, "container frame count", frame_count)
        self.stream.seek(0) # reset stream to frame 0

        return frame_index or frame_count

    def set_file(self, path):
        self.file = av.open(path)
        self.stream = self.file.streams.get(video=0)[0]#next(self.file.decode(video=0))
        self.rate = get_frame_rate(self.stream)
        self.time_base = float(self.stream.time_base)


        index, first_frame = next(self.next_frame())
        self.stream.seek(self.stream.start_time)

        # find the pts of the first frame
        index, first_frame = next(self.next_frame())

        if self.pts_seen:
            pts = first_frame.pts
        else:
            pts = first_frame.dts

        self.start_time = pts or first_frame.dts

        print("First pts", pts, self.stream.start_time, first_frame)

        #self.nb_frames = get_frame_count(self.file, self.stream)
        self.nb_frames = self.get_frame_count()

        self.update_frame_range.emit(self.nb_frames)