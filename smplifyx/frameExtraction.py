import os
import numpy as np
import cv2 as cv

video_path = 'data/ill.mp4'
out_path = 'data/ill/images/'

# ################### 单个视频抽取帧 ################################
def main():
    # 导入视频文件，参数：0 自带摄像头，1 USB摄像头，为文件名时读取视频文件
    video_caputre = cv.VideoCapture(video_path)

    # 获取读入视频的参数
    fps = video_caputre.get(cv.CAP_PROP_FPS)
    width = video_caputre.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv.CAP_PROP_FRAME_HEIGHT)

    print("fps:", fps)
    print("width:", width)
    print("height:", height)

    # 定义截取尺寸,后面定义的每帧的h和w要于此一致，否则视频无法播放
    size = (int(width), int(height))

    # 读取视频帧，然后写入文件并在窗口显示
    success, frame_src = video_caputre.read()
    c = 1
    index = 0
    while success and not cv.waitKey(1) == 27:  # 读完退出或者按下 esc 退出
        if index > 60*fps and index< 70*fps: 
            frame_target = frame_src
            cv.imwrite(out_path + str(c) + ".jpg", frame_target)
            c += 1
        # 不断读取
        success, frame_src = video_caputre.read()
        index += 1

    video_caputre.release()
    print("完成")



if __name__ == "__main__":
    main()