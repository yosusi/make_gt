import torch
import torch.nn as nn


def manual_pruning_07(makemodel):
    makemodel.input_block[1] = nn.Conv2d(3, 20, kernel_size=(7, 7), stride=(2, 2), bias=False)
    makemodel.input_block[2] = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    makemodel.down_blocks[0][0].conv1 = nn.Conv2d(20, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].conv2 = nn.Conv2d(64, 20, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn2 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].conv3 = nn.Conv2d(20, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].downsample[0] = nn.Conv2d(20, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].downsample[1] = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[0][1].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][1].conv2 = nn.Conv2d(64, 20, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn2 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][1].conv3 = nn.Conv2d(20, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[0][2].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][2].conv2 = nn.Conv2d(64, 20, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn2 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][2].conv3 = nn.Conv2d(20, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][0].conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][0].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].conv2 = nn.Conv2d(128, 39, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[1][0].bn2 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].conv3 = nn.Conv2d(39, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][0].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[1][0].downsample[1] = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][1].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][1].conv2 = nn.Conv2d(128, 39, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn2 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][1].conv3 = nn.Conv2d(39, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][2].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][2].conv2 = nn.Conv2d(128, 39, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn2 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][2].conv3 = nn.Conv2d(39, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][3].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][3].conv2 = nn.Conv2d(128, 39, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn2 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][3].conv3 = nn.Conv2d(39, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][0].conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][0].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[2][0].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][0].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[2][0].downsample[1] = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][1].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][1].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][1].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][2].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][2].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][2].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][3].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][3].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][3].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][4].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][4].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][4].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][5].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][5].conv2 = nn.Conv2d(256, 77, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn2 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][5].conv3 = nn.Conv2d(77, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][0].conv1 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][0].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].conv2 = nn.Conv2d(512, 154, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[3][0].bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].conv3 = nn.Conv2d(154, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][0].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[3][0].downsample[1] = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][1].conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][1].conv2 = nn.Conv2d(512, 154, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][1].conv3 = nn.Conv2d(154, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][2].conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][2].conv2 = nn.Conv2d(512, 154, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][2].conv3 = nn.Conv2d(154, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.bridge.bridge[0].conv = nn.Conv2d(1024, 308, kernel_size=(3, 3), stride=(1, 1))
    makemodel.bridge.bridge[0].bn = nn.BatchNorm2d(308, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.bridge.bridge[1].conv = nn.Conv2d(308, 308, kernel_size=(3, 3), stride=(1, 1))
    makemodel.bridge.bridge[1].bn = nn.BatchNorm2d(308, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[0].upsample = nn.ConvTranspose2d(308, 512, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[0].conv_block_1.conv = nn.Conv2d(1024, 154, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[0].conv_block_1.bn = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[0].conv_block_2.conv = nn.Conv2d(154, 154, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[0].conv_block_2.bn = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[1].upsample = nn.ConvTranspose2d(154, 256, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[1].conv_block_1.conv = nn.Conv2d(512, 77, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[1].conv_block_1.bn = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[1].conv_block_2.conv = nn.Conv2d(77, 77, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[1].conv_block_2.bn = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[2].upsample = nn.ConvTranspose2d(77, 128, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[2].conv_block_1.conv = nn.Conv2d(256, 39, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[2].conv_block_1.bn = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[2].conv_block_2.conv = nn.Conv2d(39, 39, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[2].conv_block_2.bn = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[3].upsample = nn.ConvTranspose2d(39, 64, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[3].conv_block_1.conv = nn.Conv2d(84, 20, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[3].conv_block_1.bn = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[3].conv_block_2.conv = nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[3].conv_block_2.bn = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[4].upsample = nn.ConvTranspose2d(20, 32, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[4].conv_block_1.conv = nn.Conv2d(35, 10, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[4].conv_block_1.bn = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[4].conv_block_2.conv = nn.Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[4].conv_block_2.bn = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.out = nn.Conv2d(10, 5, kernel_size=(1, 1), stride=(1, 1))
    
    return makemodel


def manual_pruning_05(makemodel):
    makemodel.input_block[1] = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), bias=False)
    makemodel.input_block[2] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    makemodel.down_blocks[0][0].conv1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].conv3 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][0].downsample[0] = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][0].downsample[1] = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[0][1].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][1].conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][1].conv3 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][1].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[0][2].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][2].conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[0][2].conv3 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[0][2].bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][0].conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][0].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[1][0].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][0].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[1][0].downsample[1] = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][1].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][1].conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][1].conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][1].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][2].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][2].conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][2].conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][2].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[1][3].conv1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][3].conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[1][3].conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[1][3].bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][0].conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][0].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[2][0].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][0].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[2][0].downsample[1] = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][1].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][1].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][1].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][1].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][2].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][2].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][2].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][2].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][3].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][3].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][3].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][3].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][4].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][4].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][4].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][4].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[2][5].conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][5].conv2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[2][5].conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[2][5].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][0].conv1 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][0].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].conv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
    makemodel.down_blocks[3][0].bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].conv3 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][0].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
    makemodel.down_blocks[3][0].downsample[1] = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][1].conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][1].conv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][1].conv3 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][1].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.down_blocks[3][2].conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][2].conv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    makemodel.down_blocks[3][2].conv3 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    makemodel.down_blocks[3][2].bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.bridge.bridge[0].conv = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1))
    makemodel.bridge.bridge[0].bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.bridge.bridge[1].conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
    makemodel.bridge.bridge[1].bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[0].upsample = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[0].conv_block_1.conv = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[0].conv_block_1.bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[0].conv_block_2.conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[0].conv_block_2.bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[1].upsample = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[1].conv_block_1.conv = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[1].conv_block_1.bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[1].conv_block_2.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[1].conv_block_2.bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[2].upsample = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[2].conv_block_1.conv = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[2].conv_block_1.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[2].conv_block_2.conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[2].conv_block_2.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[3].upsample = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[3].conv_block_1.conv = nn.Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[3].conv_block_1.bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[3].conv_block_2.conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[3].conv_block_2.bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[4].upsample = nn.ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))

    makemodel.up_blocks[4].conv_block_1.conv = nn.Conv2d(35, 16, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[4].conv_block_1.bn = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.up_blocks[4].conv_block_2.conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    makemodel.up_blocks[4].conv_block_2.bn = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    makemodel.out = nn.Conv2d(16, 5, kernel_size=(1, 1), stride=(1, 1))
    
    return makemodel