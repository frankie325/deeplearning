import os
from sklearn.model_selection import train_test_split
import cv2

# 处理UCF101数据集


# 生成标签txt文件
def label_text_write(ori_data_path, out_label_path):
    folder = ori_data_path
    fnames, labels = [], []
    # list = os.listdir(folder)
    # print(list) # UCF-101数据集的文件夹列表 ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
    for label in sorted(os.listdir(folder)):
        for frame in os.listdir(os.path.join(folder, label)):
            fnames.append(os.path.join(folder, label, frame))
            labels.append(label)
    # print(fnames)
    # print(labels)
    label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
    # print(label2index)

    if not os.path.exists(out_label_path + "/labels.txt"):
        os.makedirs(out_label_path)
        with open(out_label_path + "/labels.txt", "w") as f:
            for id, label in enumerate(sorted(label2index)):
                f.writelines(str(id + 1) + " " + label + "\n")


def process_video(ori_data_path, video, action_name, save_dir):
    resize_height = 128
    resize_width = 171

    # 视频文件名称
    video_filename = video.split(".")[0]
    # 视频路径
    video_path = os.path.join(save_dir, video_filename)
    if not os.path.exists(video_path):
        os.mkdir(video_path)

    capture = cv2.VideoCapture(os.path.join(ori_data_path, action_name, video))
    # print(capture)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(frame_count, frame_width, frame_height)

    # 抽帧频率为 4，意味着默认每隔 4 帧取 1 帧
    # 但如果视频帧总数小于等于 16，则抽帧频率减 1
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while count < frame_count and retaining:
        retaining, frame = capture.read()

        # 跳过读取失败的帧
        if not retaining and frame is None:
            continue
        
        # 抽取视频帧
        if count % EXTRACT_FREQUENCY == 0:
            if frame_height != resize_height or frame_width != resize_width:
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(
                filename=os.path.join(
                    save_dir, video_filename, "0000{}.jpg".format(str(i))
                ),
                img=frame,
            )
            i += 1
        count += 1

    capture.release()


#
def preprocess(ori_data_path, output_data_path):
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
        os.mkdir(os.path.join(output_data_path, "train"))
        os.mkdir(os.path.join(output_data_path, "valid"))
        os.mkdir(os.path.join(output_data_path, "test"))

    # 获取UCF101数据集的文件夹列表
    list = os.listdir(ori_data_path)
    for foldername in list:
        folderPath = os.path.join(ori_data_path, foldername)
        # 获取每个类别文件夹下的视频文件名称
        video_files = [name for name in os.listdir(folderPath)]
        # print(video_files)
        # 每个视频目录下都划分数据集
        train_and_valid, test = train_test_split(
            video_files, test_size=0.2, random_state=42
        )
        train, valid = train_test_split(train_and_valid, test_size=0.2, random_state=42)
        # print(train)
        # print(valid)
        # print(test)
        # 创建train、valid、test文件夹
        train_dir = os.path.join(output_data_path, "train", foldername)
        valid_dir = os.path.join(output_data_path, "valid", foldername)
        test_dir = os.path.join(output_data_path, "test", foldername)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for video in train:
            process_video(ori_data_path, video, foldername, train_dir)
        for video in valid:
            process_video(ori_data_path, video, foldername, valid_dir)
        for video in test:
            process_video(ori_data_path, video, foldername, test_dir)
        print("{}划分完成".format(foldername))
    print("所有数据划分完成")


if __name__ == "__main__":
    # path = os.path.abspath("./data/UCF-101")
    # print(path)
    ori_data_path = os.path.abspath("./practice/3DCNN/data/UCF-101")
    out_label_path = os.path.abspath("./practice/3DCNN/data/labels")
    output_data_path = os.path.abspath("./practice/3DCNN/data")
    # 生成标签文档
    # label_text_write(ori_data_path, out_label_path)

    # 将数据划分成训练集、验证集和测试集
    preprocess(ori_data_path, output_data_path)
