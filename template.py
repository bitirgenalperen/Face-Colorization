from cProfile import label
from tabnanny import check
from turtle import color, forward
import hw3utils
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchsummary import summary
from utils import read_image
import sys

batch_size = 16
max_num_epoch = 100
hps = {'lr': 0.0005}

# ---- options ----
# set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
DEVICE_ID = 'cuda:0'
LOG_DIR = 'checkpoints'
OUT_DIR = 'out'
# set True to visualize input, prediction and the output from the last batch
VISUALIZE = False
LOAD_CHKPT = False
base_parameters = [['base', 2, 5, 4, 0.007], ['base', 4, 5, 4, 0.007], ['base', 2, 3, 4, 0.007],
                   ['base', 2, 5, 8, 0.007], ['base', 2, 5, 4, 0.070], ['base', 2, 5, 4, 0.0007]]
improved_parameters = [['improved', 2, 5, 8, 0.07, True, False], ['improved', 2, 5, 8, 0.07, False, True],
                       ['improved', 2, 5, 8, 0.07, True, True], ['improved', 2, 5, 16, 0.07, False, True]]
# --- imports ---
torch.multiprocessing.set_start_method('spawn', force=True)
# ---- utility functions -----


def get_loaders(batch_size, device):
    data_root = 'dataset'
    train_set = hw3utils.HW3ImageFolder(
        root=os.path.join(data_root, 'train'), device=device)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    train_shuffle_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(
        root=os.path.join(data_root, 'val'), device=device)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_shuffle_loader, train_loader, val_loader


class BaseNet(nn.Module):
    def __init__(self, layer_count, kern_size, kern_count):
        super(BaseNet, self).__init__()
        self.layer_count = layer_count
        self.convi = nn.Conv2d(
            1, kern_count, kernel_size=kern_size, padding=kern_size//2)
        self.convm = nn.Conv2d(
            kern_count, kern_count, kernel_size=kern_size, padding=kern_size//2)
        self.convl = nn.Conv2d(
            kern_count, 3, kernel_size=kern_size, padding=kern_size//2)
        self.conv_single = nn.Conv2d(
            1, 3, kernel_size=kern_size, padding=kern_size//2)
        self.activation = F.relu

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        if(self.layer_count == 1):
            x = self.conv_single(grayscale_image)

        elif (self.layer_count > 1):
            x = self.convi(grayscale_image)
            x = self.activation(x)
            for i in range(self.layer_count-2):
                x = self.convm(x)
                x = self.activation(x)
            x = self.convl(x)
        return x


class ImprovedNet(nn.Module):
    def __init__(self, layer_count, kern_size, kern_count, batch_norm=False, last_layer_activation=False):
        super(ImprovedNet, self).__init__()
        self.layer_count = layer_count
        self.batch_norm = batch_norm
        self.last_layer_activation = last_layer_activation
        self.convi = nn.Conv2d(
            1, kern_count, kernel_size=kern_size, padding=kern_size//2)
        self.convm = nn.Conv2d(
            kern_count, kern_count, kernel_size=kern_size, padding=kern_size//2)
        self.convl = nn.Conv2d(
            kern_count, 3, kernel_size=kern_size, padding=kern_size//2)
        self.conv_single = nn.Conv2d(
            1, 3, kernel_size=kern_size, padding=kern_size//2)
        self.batch = nn.BatchNorm2d(kern_count)
        self.batchl = nn.BatchNorm2d(3)
        self.activation = F.relu
        self.activation2 = torch.tanh

    def forward(self, grayscale_image):
        if(self.layer_count == 1):
            x = self.conv_single(grayscale_image)
        elif(self.layer_count > 1):
            x = self.convi(grayscale_image)
            if(self.batch_norm):
                x = self.batch(x)
            x = self.activation(x)
            for i in range(self.layer_count-2):
                x = self.convm(x)
                if(self.batch_norm):
                    x = self.batch(x)
                x = self.activation(x)

            x = self.convl(x)
            if(self.batch_norm):
                x = self.batchl(x)
            if(self.last_layer_activation):
                x = self.activation2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),  # 8 80 80
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                 # 8 40 40
            nn.Conv2d(8, 4, 3, stride=2, padding=1),    # 4 20 20
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)                   # 4 10 10
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 3, 2, stride=1),
            nn.Tanh()
        )

    def forward(self, grayscale_image):
        x = self.encoder(grayscale_image)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self, kern_size=5):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=kern_size, padding=kern_size//2)
        self.conv2 = nn.Conv2d(
            8, 12,  kernel_size=kern_size, padding=kern_size//2)
        self.conv3 = nn.Conv2d(
            12, 18,  kernel_size=kern_size, padding=kern_size//2)
        self.conv4 = nn.Conv2d(
            18, 16,  kernel_size=kern_size, padding=kern_size//2)
        self.conv5 = nn.Conv2d(
            28, 12, kernel_size=kern_size, padding=kern_size//2)
        self.conv6 = nn.Conv2d(
            20, 3,  kernel_size=kern_size, padding=kern_size//2)
        self.activation = F.relu

    def forward(self, grayscale_image):
        x0 = self.conv1(grayscale_image)
        x0 = self.activation(x0)
        x1 = self.conv2(x0)
        x1 = self.activation(x1)
        x2 = self.conv3(x1)
        x2 = self.activation(x2)
        x3 = self.conv4(x2)
        x3 = self.activation(x3)
        x4 = self.conv5(torch.cat([x3, x1], dim=1))
        x4 = self.activation(x4)
        x = self.conv6(torch.cat([x4, x0], dim=1))
        return x


class WeirdNet(nn.Module):
    def __init__(self):
        super(WeirdNet, self).__init__()
        self.activation = F.leaky_relu
        self.activation2 = F.tanh
        self.conv1_4 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv1_8 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv1_16 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv4_4 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self.conv8_8 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.conv16_16 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.conv12 = nn.Conv2d(12, 6, kernel_size=5, padding=2)
        self.conv20 = nn.Conv2d(20, 10, kernel_size=5, padding=2)
        self.conv24 = nn.Conv2d(24, 12, kernel_size=5, padding=2)
        self.convl = nn.Conv2d(28, 8, kernel_size=5, padding=2)
        self.convl2 = nn.Conv2d(8, 3, kernel_size=5, padding=2)

    def forward(self, grayscale_image):
        x00 = self.activation(self.conv1_4(grayscale_image))
        x01 = self.activation(self.conv1_8(grayscale_image))
        x02 = self.activation(self.conv1_16(grayscale_image))
        x10 = self.activation(self.conv4_4(x00))
        x11 = self.activation(self.conv8_8(x01))
        x12 = self.activation(self.conv16_16(x02))
        x20 = self.activation(self.conv12(torch.cat([x10, x11], dim=1)))
        x21 = self.activation(self.conv20(torch.cat([x10, x12], dim=1)))
        x22 = self.activation(self.conv24(torch.cat([x11, x12], dim=1)))
        x30 = self.convl(torch.cat([x20, x21, x22], dim=1))
        x40 = self.convl2(x30)
        return x40


def train_validate(model_details):
    # ---- training code -----
    device = torch.device(DEVICE_ID)
    best_acc = 0
    best_acc_idx = None
    print('device: ' + str(device))
    if (model_details[0] == 'base'):
        net = BaseNet(model_details[1], model_details[2],
                      model_details[3]).to(device=device)
        summary(net, (1, 80, 80))
    elif (model_details[0] == 'improved'):
        net = ImprovedNet(model_details[1], model_details[2],
                          model_details[3], model_details[5], model_details[6]).to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'autoencoder'):
        net = AutoEncoder().to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'unet'):
        net = UNet().to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'weird'):
        net = WeirdNet().to(device=device)
        summary(net, (1, 80, 80))
    else:
        print("UNRECOGNIZED ARCHITECTURE")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.07)
    train_loader, train_ordered, val_loader = get_loaders(batch_size, device)

    if LOAD_CHKPT:
        print('loading the model from the checkpoint')
        # model.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

    train_losses = []
    validation_losses = []
    train_acc = []
    validation_acc = []
    file_name = 'estimations__' + str(model_details)
    print('training begins')
    for epoch in range(max_num_epoch):

        for iteri, data in enumerate(train_loader, 0):
            # inputs: low-resolution images, targets: high-resolution images.
            inputs, targets = data
            optimizer.zero_grad()  # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            # print loss

            if (iteri == 0) and VISUALIZE:
                hw3utils.visualize_batch(inputs, preds, targets)
        estimations_train = np.zeros(shape=(5001, 80, 80, 3))
        # train loss and acc
        with torch.no_grad():
            running_train_loss = 0.0
            s1 = 0
            i = 0
            for idx, train_data in enumerate(train_ordered, 0):
                train_i, train_t = train_data
                train_predict = net(train_i)
                for t in train_predict.data:
                    k = transforms.functional.to_pil_image(t.cpu()/2+0.5)
                    estimations_train[i, :, :, :] = k
                    i += 1
                train_loss = criterion(train_predict, train_t)
                running_train_loss += train_loss.item()
                s1 += 1
            a, l = accuracy_calc(
                estimations_train, "train"), running_train_loss/s1
            print("Epoch: " + str(epoch+1) +
                  " :: Train Loss: " + str(l) + " :: Train Acc: " + str(a))
            train_losses.append(l)
            train_acc.append(a)

        # note: you most probably want to track the progress on the validation set as well (needs to be implemented)
        estimations_validation = np.zeros(shape=(2000, 80, 80, 3))
        with torch.no_grad():
            running_val_loss = 0.0
            s2 = 0
            i = 0
            for idx, val_data in enumerate(val_loader, 0):
                val_i, val_t = val_data
                val_predict = net(val_i)
                for p in val_predict.data:
                    k = transforms.functional.to_pil_image(p.cpu()/2+0.5)
                    estimations_validation[i, :, :, :] = k
                    i += 1
                val_loss = criterion(val_predict, val_t)
                running_val_loss += val_loss.item()
                s2 += 1
            a, l = accuracy_calc(estimations_validation,
                                 "validation"), running_val_loss/s2
            print("Epoch: " + str(epoch+1) +
                  " :: Validation Loss: " + str(l) + " :: Validation Acc: " + str(a))
            validation_losses.append(l)
            validation_acc.append(a)

        if(validation_acc[-1] > best_acc):
            best_acc = validation_acc[-1]
            best_acc_idx = epoch
            print('Saving the model, end of epoch %d' % (epoch+1))
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            f_name = os.path.join(LOG_DIR, str(model_details))
            torch.save(net.state_dict(), f"{f_name}.pt")
            hw3utils.visualize_batch(inputs, preds, targets,
                                     f"{f_name}.png")

        # stopping the epochs
        # progress_controller(validation_losses) or
        if(progress_controller(validation_losses) or epoch == 99):
            x = np.arange(0, len(train_losses))
            fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
            ax[0].plot(train_losses, color='blue', label='Training Loss')
            ax[0].plot(validation_losses, color='red', label='Validation Loss')
            ax[0].set_ylim(
                [0, max(np.max(validation_losses), np.max(train_losses)) + 0.01])
            ax[0].set_title("Training and Validation Loss")
            ax[0].legend()
            ax[1].plot(train_acc, color='black', label='Training acc')
            ax[1].plot(validation_acc, color='green', label='Validation acc')
            ax[1].set_ylim(
                [0, max(np.max(validation_acc), np.max(train_acc)) + 0.01])
            ax[1].set_title("Training and Validation Accuracy")
            ax[1].legend()
            plt.savefig(f"{file_name}.png")

            file_name += '.npy'

            print('Finished Training :: ', best_acc_idx+1)
            print("Accuracy: ", validation_acc[best_acc_idx])
            print("Losses: ", validation_losses[best_acc_idx])
            return (validation_acc[best_acc_idx],  validation_losses[best_acc_idx], best_acc_idx+1)


# checks the validation losses and decides if the training stop or continue
def progress_controller(loss, patience=10, delta=0.00005):
    if(len(loss) <= patience):
        return False
    else:
        difference = np.asarray(loss[-patience:])
        margin_difference = np.abs(difference[1:] - difference[:-1])
        if (np.all(margin_difference < delta)):
            return True
        else:
            return False


def accuracy_calc(estimations, name_file):
    name_file += "_names.txt"
    with open(name_file, "r") as f:
        files = f.readlines()

    acc = 0
    for i, file in enumerate(files):
        cur = read_image(file.rstrip()).reshape(-1).astype(np.int64)
        est = estimations[i].reshape(-1).astype(np.int64)
        cur_acc = (np.abs(cur - est) < 12).sum() / cur.shape[0]
        acc += cur_acc
    acc /= len(files)
    return acc


def test(model_details=['improved', 2, 5, 16, 0.07, False, True]):
    device = torch.device(DEVICE_ID)
    if (model_details[0] == 'base'):
        net = BaseNet(model_details[1], model_details[2],
                      model_details[3]).to(device=device)
        summary(net, (1, 80, 80))
    elif (model_details[0] == 'improved'):
        net = ImprovedNet(model_details[1], model_details[2],
                          model_details[3], model_details[5], model_details[6]).to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'autoencoder'):
        net = AutoEncoder().to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'unet'):
        net = UNet().to(device=device)
        summary(net, (1, 80, 80))
    elif(model_details[0] == 'weird'):
        net = WeirdNet().to(device=device)
        summary(net, (1, 80, 80))
    else:
        print("UNRECOGNIZED ARCHITECTURE")

    path = str(model_details) + ".pt"
    checkpoint_path = os.path.join(LOG_DIR, path)
    net.load_state_dict(torch.load(checkpoint_path))

    # get some 100 the test images
    test_root = os.path.join('dataset', 'test_inputs')
    test_set = hw3utils.HW3ImageFolder(root=test_root, device=device)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0)

    cur_imgs = os.listdir(os.path.join(test_root, 'images'))

    np.random.seed(0)
    aa = np.unique(np.random.randint(low=0, high=2000, size=120))[:100]
    selective_idx = np.sort(aa)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    estimations_test = np.zeros(shape=(100, 80, 80, 3))
    with torch.no_grad():
        i = 0
        for idx, test_data in enumerate(test_loader, 0):

            if(idx == selective_idx[i]):

                test_i, _ = test_data
                test_predict = net(test_i)

                for t in test_predict.data:
                    k = transforms.functional.to_pil_image(t.cpu()/2+0.5)
                    estimations_test[i, :, :, :] = np.asarray(
                        k).reshape((80, 80, -1))
                i += 1
                if(i == 100):
                    break
    np.save('estimations_test.npy', estimations_test)
    for i in range(100):
        with open("test_names.txt", 'a') as fp:
            fp.write(os.path.join(test_root, cur_imgs[selective_idx[i]]))
            fp.write('\n')
        plt.imsave(os.path.join(
            OUT_DIR, cur_imgs[selective_idx[i]]), estimations_test[i].astype(np.uint8))


def custom_process_selector(args):
    # [layer_count, kernel_size, kernel_count,  learning_rate, batch_norm, tanh]
    if(args[0] == 'custom'):
        args[-2:] = [True if i in ['true', 'True'] else False for i in args[-2:]]
        args[1:-2] = [int(args[1]), int(args[2]), int(args[3]), float(args[4])]
        print(args)
        train_validate(['improved', args[1], args[2], args[3],
                       args[4], args[5], args[6]])
        if(args[7] == 'true'):
            test(['improved', args[1], args[2], args[3], args[4], args[5], args[6]])

    elif(args[0] == 'base-all'):
        for each in base_parameters:
            print(each)
            a, l, e = train_validate(each)
            p = "layer_count: " + str(each[1]) + " :: kernel_size: " + str(
                each[2]) + " :: kernel_count: " + str(each[3]) + " :: learning_rate: " + str(each[4])
            r = "epoch: " + str(e) + " :: acc: " + \
                str(a) + " :: loss: " + str(l)
            with open('base.txt', 'a') as fp:
                fp.write("*********************\n")
                fp.write(p)
                fp.write("\n")
                fp.write(r)
                fp.write("\n")
                fp.write("*********************\n")
    elif(args[0] == 'base-best'):
        train_validate(['base', 2, 5, 8, 0.07])

    elif(args[0] == 'improved-all'):
        for each in improved_parameters:
            a, l, e = train_validate(each)
            p = "layer_count: " + str(each[1]) + " :: kernel_size: " + str(
                each[2]) + " :: kernel_count: " + str(each[3]) + " :: learning_rate: " + str(each[4]) + " :: batchnorm: " + str(each[5]) + " :: tanh: " + str(each[6])
            r = "epoch: " + str(e) + " :: acc: " + \
                str(a) + " :: loss: " + str(l)
            with open('improved.txt', 'a') as fp:
                fp.write("*********************\n")
                fp.write(p)
                fp.write("\n")
                fp.write(r)
                fp.write("\n")
                fp.write("*********************\n")
    elif(args[0] == 'improved-best'):
        # Using best config. resultant test image output will be held in 'out' directory
        train_validate(['improved', 2, 5, 16, 0.07, False, True])
        if(args[1] == 'true'):
            test(model_details=['improved', 2, 5, 16, 0.07, False, True])
    ### additional works ###
    elif(args[0] == 'autoencoder'):
        train_validate(['autoencoder'])

        if(args[1] == 'true'):
            test(model_details=['autoencoder'])

    elif(args[0] == 'unet'):
        train_validate(['unet'])
        if(args[1] == 'true'):
            test(model_details=['unet'])

    elif(args[0] == 'weird'):
        train_validate(['weird'])
        if(args[1] == 'true'):
            test(model_details=['weird'])

    elif(args[0] == 'test-best-only'):
        test(model_details=['improved', 2, 5, 16, 0.07, False, True])


if __name__ == '__main__':
    if(len(sys.argv) >= 2):
        print(sys.argv[1:])
        custom_process_selector(sys.argv[1:])
    else:
        print("CANNOT FIND ANY GIVEN PARAMETER")
