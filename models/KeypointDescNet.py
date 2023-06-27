import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointDescriptorNet(nn.Module):
    def __init__(self, original_model, mid_channels=1024, desc_len=256):
        super(KeypointDescriptorNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(mid_channels, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.keypoint_desc_regressor = nn.Sequential(
            nn.Linear(2048, desc_len),
            nn.Sigmoid(),
        )
        self.modelName = 'resnet'

        print(self)

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.keypoint_desc_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        f = self.features(inpt)
        # print(f.size())
        f = f.view(f.size(0), -1)
        # print(f.size())
        y = self.regressor(f)
        desc = self.keypoint_desc_regressor(y)
        desc = F.normalize(desc, dim=1, p=2)

        return desc
