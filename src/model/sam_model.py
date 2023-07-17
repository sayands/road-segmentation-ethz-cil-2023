from transformers import SamModel
from torch import nn


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
        self.up_scale = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=4, padding=0,
                                           output_padding=0, dilation=3)
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def forward(self, inputs):
        x = self.up_scale(inputs)
        # remove some unused dimensions
        prediction = self.model(x, multimask_output=False)
        prediction = prediction.pred_masks[0][0][0]
        return prediction
