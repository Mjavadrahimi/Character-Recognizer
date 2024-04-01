import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 26)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, input: torch.Tensor, train=False):
        input = input.view(-1, 1, 28, 28)
        # size = X * 28*28 * 1

        output = self.relu(self.pool(self.conv1(input)))
        # size = X * 12*12 * 6
        output = self.relu(self.pool(self.conv2(output)))
        # size = X * 4*4 * 16

        output = output.view(-1, 16 * 4 * 4)  # flattening
        # size = X * 256

        if train: output = self.dropout(output)
        output = self.l1(output)
        output = self.relu(output)

        if train: output = self.dropout(output)
        output = self.l2(output)
        output = self.relu(output)

        if train: output = self.dropout(output)
        output = self.l3(output)
        output = self.sig(output)
        return output


def image_show(image: Image.Image, title=''):
    plt.imshow(image, cmap='grey')
    if title:
        plt.title(title)
    plt.show()


CODE_TO_CHAR = {i: chr(ord('A') + i) for i in range(26)}
CHAR_TO_CODE = {chr(ord('A') + i): i for i in range(26)}


class CharacterRecognize:
    def __init__(self, PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(256, 120, 80).to(self.device)
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def image_to_tensor(self, image: Image.Image):  # MinMax_scale_image #grey_scale #to_tensor #to_device
        transform = transforms.ToTensor()
        image_gery = image.convert("L")
        torch_image = transform(image_gery).to(self.device)
        return torch_image

    def predict(self, image: Image.Image, top_k=None):
        torch_image = self.image_to_tensor(image)
        pred = self.model(torch_image)
        if top_k:
            top_values, top_indices = torch.topk(pred, k=top_k, sorted=True)
            output = []
            for value, index in zip(top_values[0], top_indices[0]):
                output.append((CODE_TO_CHAR[index.item()], value.item()))
            return output
        else:
            return CODE_TO_CHAR[torch.argmax(pred).item()], torch.max(pred).item()


if __name__ == '__main__':
    PATH = 'character_recognizer_model.pth'
    obj = CharacterRecognize(PATH)
    test_size = 12
    manual_test_y = ('A', 'B', 'C', 'D', 'E', 'Z', 'W', 'U', 'V', 'O', 'Q', 'S')
    for i in range(test_size):
        image_path = f'../data/manual_test/{i}.png'
        image = Image.open(image_path)
        pred_3 = obj.predict(image, top_k=3)
        pred = obj.predict(image)
        image_show(image, title=f'pred = {pred[0]} | actual = {manual_test_y[i]}')
        print(pred_3, '| actual', manual_test_y[i])
