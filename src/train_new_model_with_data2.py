import time
from torch import nn, optim
import dsntnn
from keypoints_Net import CoordRegression

from data_process import *

torch.manual_seed(12345)


def train_new_data_with_model():
    model = CoordRegression(n_locations=8)
    optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4, alpha=0.9)
    model = torch.nn.DataParallel(model).cuda()

    from data_process_landmarks_hw import dataloader
    # 训练集
    dataloader = dataloader

    for epoch in range(10):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, 10))
        train_loss = 0.0
        train_loss_cord = []

        # forward pass
        model.train()

        # 训练
        for i_batch, data in enumerate(dataloader):
            img, landmarks = data
            img = torch.tensor(img, dtype=torch.float32)
            img = img.to(device)
            landmarks = torch.tensor(landmarks / 64.0, dtype=torch.float32)
            landmarks = landmarks.to(device)
            # print("Ground-truth:", gt_hmap.shape)
            optimizer.zero_grad()
            # forward pass
            coords, heatmaps = model(img)
            # per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, landmarks)
            # print("predict coords", coords, landmarks)
            # per-location regulation losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, landmarks, sigma_t=1.0)
            # combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()

            # update model parameters with RMSprop
            optimizer.step()

            train_loss_cord.append(loss)

            if i_batch % 20 == 19:
                print(loss, euc_losses, reg_losses)
                # break
            # print(loss)
        torch.save(model, 'models/' + 'landmarks' + '_model_new_data_8' + str(epoch) + '.pt')
        print(train_loss_cord)


def main():
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoordRegression(n_locations=8)

    summary(model.cuda(), (3, 300, 300))


if __name__ == "__main__":
    # 测试打印模型
    # main()

    train_new_data_with_model()
