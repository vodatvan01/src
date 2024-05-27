import  random
import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
import torch
from torchvision import transforms



class ComposeTrain(object):
    def __init__(self, trans_to_rgb=True):
        self.trans_to_rgb = trans_to_rgb

        # Khởi tạo các bước xử lý ảnh
        self.load_image = LoadImage(trans_to_rgb=trans_to_rgb)
        self.resize_by_short = ResizeByShort()
        self.random_crop = RandomCrop()
        self.flip_image = FlipImage()
        self.gen_trimap = GenTrimap()
        self.resize = Resize()
        self.normalize_tensor = NormalizeTensor()

    def __call__(self, data):
        image_path, mask_path = data[0], data[1]
        # Load ảnh và mask từ đường dẫn
        image, alpha = self.load_image([image_path, mask_path])

        # Resize ảnh và mask theo kích thước ngắn nhất
        image, alpha = self.resize_by_short([image, alpha])

        # Áp dụng random crop
        image, alpha = self.random_crop([image, alpha])

        # Flip ảnh và mask (nếu có)
        image, alpha = self.flip_image([image, alpha])

        # Tạo trimap từ alpha mask
        trimap = self.gen_trimap(alpha)

        # Resize lại ảnh, alpha mask và trimap
        image, alpha, trimap = self.resize([image, alpha, trimap])

        # Chuẩn hóa tensor của ảnh và alpha mask
        image, alpha, trimap = self.normalize_tensor([image, alpha, trimap])

        return image, alpha, trimap


class ComposeValidation(object):
    def __init__(self, trans_to_rgb=True):
        self.trans_to_rgb = trans_to_rgb

        # Khởi tạo các bước xử lý ảnh
        self.load_image = LoadImage(trans_to_rgb=trans_to_rgb)
        self.resize_by_short = ResizeByShort()
        self.gen_trimap = GenTrimap()
        self.resize = Resize()
        self.normalize_tensor = NormalizeTensor()

    def __call__(self, data):
        image_path, mask_path = data[0], data[1]
        # Load ảnh và mask từ đường dẫn
        image, alpha = self.load_image([image_path, mask_path])

        # Resize ảnh và mask theo kích thước ngắn nhất
        image, alpha = self.resize_by_short([image, alpha])

        # Tạo trimap từ alpha mask
        trimap = self.gen_trimap(alpha)

        # Resize lại ảnh, alpha mask và trimap
        image, alpha, trimap = self.resize([image, alpha, trimap])

        # Chuẩn hóa tensor của ảnh và alpha mask
        image, alpha, trimap = self.normalize_tensor([image, alpha, trimap])

        return image, alpha, trimap
    

class LoadImage(object):
    def __init__(self, trans_to_rgb=True):
        self.trans_to_rgb = trans_to_rgb
 
    def __call__(self, data): #data = ['image_path', 'mask_path']
        if isinstance(data[0], str): # data[0] = 'image_path'
            image = cv2.imread(data[0])
        if isinstance(data[1], str): # data[1] = 'mask_path'
            alpha = cv2.imread(data[1], 0)

        if self.trans_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, alpha
    

class ResizeByShort(object):
    def __init__(self, ref_size=512, interp='area'):
        if not isinstance(ref_size, int):
            raise TypeError(
                "Type of `ref_size` is invalid. It should be int, but it is {}"
                .format(type(ref_size)))

        self.ref_size = ref_size
        if interp == 'area':
            self.interps = cv2.INTER_AREA
        else:
            self.interps = cv2.INTER_LINEAR

    def __call__(self, data):
        if len(data) != 2:
            raise ValueError("Input data should contain exactly two elements: image array and alpha mask array")

        image = data[0]
        alpha = data[1]

        im_h, im_w, _ = image.shape
        # print(f'im_h : {im_h} and im_w : {im_w}')

        if max(im_h, im_w) < self.ref_size or min(im_h, im_w) != self.ref_size: 
            if im_w >= im_h:
                im_rh = self.ref_size
                im_rw = int(im_w / im_h * self.ref_size)
            else:
                im_rw = self.ref_size
                im_rh = int(im_h / im_w * self.ref_size)
            
        else: # nếu im_h, im_w = 512    
            im_rh = im_h
            im_rw = im_w

        # Resize to nearest multiple of 32
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        # Resize image
        image_resized = cv2.resize(image, (im_rw, im_rh), self.interps)

        # Resize alpha mask
        alpha_resized = cv2.resize(alpha, (im_rw, im_rh), cv2.INTER_NEAREST)

        return image_resized, alpha_resized



class RandomCrop(object):
    def __init__(self, crop_size=((512,512), (768, 768), (1024, 1024))):
        if not isinstance(crop_size[0], (list, tuple)):
            crop_size = [crop_size]
        self.crop_size = crop_size

    def __call__(self, data):
        if len(data) != 2:
            raise ValueError("Input data should contain exactly two elements: image array, alpha mask array")

        image = data[0]
        alpha_mask = data[1]
        # trimap = data[2]

        # Randomly select a crop size
        crop_h, crop_w = random.choice(self.crop_size)

        img_h, img_w = image.shape[:2]

        

        # Calculate random start position for the crop
        start_h, start_w = 0,0
        if img_h > crop_h :
            start_h = np.random.randint(0, img_h - crop_h + 1)
        if img_w > crop_w :
            start_w = np.random.randint(0, img_w - crop_w + 1)

       

        # Calculate end position for the crop
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        # Crop the image
        cropped_image = image[start_h:end_h, start_w:end_w]

        # Crop the alpha mask
        cropped_alpha_mask = alpha_mask[start_h:end_h, start_w:end_w]

        return cropped_image, cropped_alpha_mask 


class FlipImage(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):

        image = data[0]
        alpha = data[1]
        flip_flag = random.random() < self.probability
        if flip_flag:
            flipped_image = cv2.flip(image,1)
            flipped_alpha = cv2.flip(alpha,1)
            return flipped_image, flipped_alpha  
        else:
            return image, alpha
        


class GenTrimap(object):
    def __init__(self, alpha_size=512, max_value=1):
        
        self.alpha_size = alpha_size
        if max_value == 255:
            self.gen_fn = self.gen_trimap_with_dilate
        else:
            self.gen_fn = self.gen_trimap_sci

    def __call__(self, alpha):
        trimap = self.gen_fn(alpha)
        return trimap

    def gen_trimap_sci(self, alpha):
        trimap = (alpha >= 0.9).astype('float32')
        not_bg = (alpha > 0).astype('float32')

        d_size = self.alpha_size // 256 * 15
        e_size = self.alpha_size // 256 * 15
        trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size))
                         - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        

        return trimap

    def gen_trimap_with_dilate(self, alpha):
        kernel_size = random.randint(15, 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
        erode = cv2.erode(fg, kernel, iterations=1)
        trimap = erode *255 + (dilate-erode)*128
        return trimap.astype(np.uint8)
    
class Resize(object):
    def __init__(self, target_size=(512, 512), interp='area'):
        if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
            raise ValueError("`target_size` should be a list or tuple containing two elements (height, width)")

        self.target_size = target_size
        if interp == 'area':
            self.interp = cv2.INTER_AREA
        else:
            self.interp = cv2.INTER_LINEAR

    def __call__(self, data):
        if len(data) != 3:
            raise ValueError("Input data should contain exactly three elements: image array, alpha mask array, and trimap array")
        image = data[0]
        alpha = data[1]
        trimap = data[2]

        
        img_h, img_w = image.shape[:2]
        if img_h == 512 and img_w == 512:
            return data
        
        # Resize image
        image_resized = cv2.resize(image, self.target_size, interpolation=self.interp)
        # Resize alpha mask
        alpha_resized = cv2.resize(alpha, self.target_size, interpolation=cv2.INTER_NEAREST)
        # Resize trimap
        trimap_resized = cv2.resize(trimap, self.target_size, interpolation=cv2.INTER_NEAREST)
        return image_resized, alpha_resized, trimap_resized

class NormalizeTensor(object):
    def __init__(self, colorMean = (0.485, 0.456, 0.406), colorStd = (0.229, 0.224, 0.225)):
        self.mean = colorMean
        self.std = colorStd
    def __call__(self,data ):
        image = data[0]
        alpha = data[1]
        trimap = data[2]

        # image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, self.mean, self.std)
        alpha = torch.from_numpy(np.array(alpha)/255.0).unsqueeze(0)
        trimap = torch.from_numpy(np.array(trimap)).unsqueeze(0)
        return image, alpha, trimap
