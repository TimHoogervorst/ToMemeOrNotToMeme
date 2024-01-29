
import glob
from PIL import Image
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

class PreProcess():
    def __init__(self, transform) -> None:
       self.transform = transform
       pass 
        
    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image, self.transform(image)

    def pre_prosses_image(self, image_path) -> None:
        _, prossessed_image = self.process_image(image_path)
        prossessed_image.save(image_path)

    def pre_prosses_data_set(self, folder_path) -> None:
        images = glob.glob(f"{folder_path}/*")
        for image in tqdm(images):
            _, prossessed_image = self.process_image(image)
            prossessed_image.save(image)
