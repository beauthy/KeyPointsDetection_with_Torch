import numpy as np
import matplotlib.pyplot as plt


def generate_hmap_mask(imageSize=(224, 224), landmarks=None):
    # (kp + background) * num
    if landmarks is None:
        landmarks = [[20, 220, 1], [100, 120, 0]]
    landmarks = np.array(landmarks)  # 切片只可以是整数
    h, w = imageSize
    hmap_mask = []
    radius = 80
    gaussMask = make_gaussian(radius, radius, 18, None)

    for landmark in landmarks:
        gthmp = np.zeros((h, w))
        if landmark[2] == -1:
            hmap_mask.append(gthmp)
            continue

        # avoid out of boundary
        top_x, top_y = max(0, int(landmark[0] - radius / 2)), max(0, int(landmark[1] - radius / 2))  # 高斯图中心点和关键点重合，左上角

        bottom_x, bottom_y = min(w, int(landmark[0] + radius / 2)), min(h, int(landmark[1] + radius / 2))  # 右下脚坐标

        top_x_offset = top_x - int(landmark[0] - radius / 2)  # 在高斯图上的偏移
        top_y_offset = top_y - int(landmark[1] - radius / 2)  # 偏移

        gthmp[top_y:bottom_y, top_x:bottom_x] = gaussMask[top_y_offset:top_y_offset + bottom_y - top_y,
                                                top_x_offset:top_x_offset + bottom_x - top_x]
        hmap_mask.append(gthmp)

    return np.array(hmap_mask)


def generate_hmap(imageSize=(224, 224), landmarks=None):
    # kpnum + background
    if landmarks is None:
        landmarks = [[20, 220, 1], [100, 120, 0]]
    landmarks = np.array(landmarks)  # 切片只可以是整数
    h, w = imageSize
    gthmp = np.zeros((h, w))

    radius = 80
    gaussMask = make_gaussian(radius, radius, 18, None)

    for landmark in landmarks:
        if landmark[2] == -1:
            continue

        # avoid out of boundary
        top_x, top_y = max(0, int(landmark[0] - radius / 2)), max(0, int(landmark[1] - radius / 2))  # 高斯图中心点和关键点重合，左上角

        bottom_x, bottom_y = min(w, int(landmark[0] + radius / 2)), min(h, int(landmark[1] + radius / 2))  # 右下脚坐标

        top_x_offset = top_x - int(landmark[0] - radius / 2)  # 在高斯图上的偏移
        top_y_offset = top_y - int(landmark[1] - radius / 2)  # 偏移

        gthmp[top_y:bottom_y, top_x:bottom_x] = gaussMask[top_y_offset:top_y_offset + bottom_y - top_y,
                                                top_x_offset:top_x_offset + bottom_x - top_x]

    return gthmp


def make_gaussian(width, height, sigma=3, center=None):
    '''
        generate 2d guassion heatmap
        反应该点是关键点的概率
    :return:
    '''

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]

    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def show_landmarks(image):
    plt.imshow(image)
    plt.pause(5)  # pause a bit so that plots are updated


if __name__ == "__main__":
    # 测试关键点生成高斯热图
    print(generate_hmap_mask().shape)

    pass
