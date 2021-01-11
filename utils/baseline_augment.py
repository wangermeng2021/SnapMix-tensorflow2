
import albumentations as A
import cv2
def resize_and_random_crop(img,resize_size,crop_size):
    img = cv2.resize(img, resize_size)
    transform = A.Compose([
        A.RandomCrop(width=crop_size[0], height=crop_size[1]),
        A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
    ])
    img = transform(image=img)['image']
    return img

def resize_and_center_crop(img,resize_size,crop_size):
    img = cv2.resize(img, resize_size)
    transform = A.Compose([
        A.CenterCrop(width=crop_size[0], height=crop_size[1]),
    ])
    img = transform(image=img)['image']
    return img