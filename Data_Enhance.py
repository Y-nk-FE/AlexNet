import os
from PIL import Image, ImageEnhance
import random
import cv2


def Random_Enhance(img):
    """
    随机数据增强，包含对数据的镜像翻转、旋转、随即尺度变换、亮度变化、锐度变化、对比度变化、色彩平衡
    :param img:
    :return:
    """
    # transpose-翻转,rotate-旋转,
    methods = [Image.Image.transpose,
               Image.Image.rotate,
               Image.Image.resize,
               ImageEnhance.Brightness,
               ImageEnhance.Sharpness,
               ImageEnhance.Contrast,
               ImageEnhance.Color]
    method = random.choice(methods)

    if method == Image.Image.transpose:
        # 左右镜像翻转、上下镜像翻转、逆时针旋转90、逆时针旋转180、逆时针旋转270
        methods = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        return img.transpose(random.choice(methods))
    elif method == Image.Image.rotate:
        # 逆时针旋转随机度数(0-360)
        return img.rotate(random.randint(0, 360))
    elif method == Image.Image.resize:
        # 宽高随机缩放原来图像的0.5-1.5倍大小
        width, height = img.size
        return img.resize(
            (random.randint(int(width * 0.5), int(width * 1.5)), random.randint(int(height * 0.5), int(height * 1.5))))
    elif method == ImageEnhance.Brightness:
        # 亮度变化
        factor = random.uniform(a=0.3, b=1.1)
        return ImageEnhance.Brightness(image=img).enhance(factor=factor)
    elif method == ImageEnhance.Sharpness:
        #
        factor = random.uniform(a=0.5, b=1.5)
        return ImageEnhance.Sharpness(image=img).enhance(factor=factor)
    elif method == ImageEnhance.Contrast:
        # 对比度变化
        factor = random.uniform(a=0.3, b=1.7)
        return ImageEnhance.Sharpness(image=img).enhance(factor=factor)
    elif method == ImageEnhance.Color:
        factor = random.uniform(a=0.0, b=1.0)
        return ImageEnhance.Color(image=img).enhance(factor=factor)


def batch_enhance_image(class_input_dir, class_output_dir, suffix):
    """
    :param class_input_dir: 原始数据文件夹（INPUT）
    :param class_output_dir: 增强文件夹（OUTPUT）
    :param suffix: 新数据编号(如果需要多轮增强就需要保证文件名不相同，故此处创建一个编号加在原始文件名后来区分)
    :return: None
    """
    # 检查输出目录是否存在，如果不存在，则创建它
    if not os.path.exists(class_output_dir):
        os.makedirs(class_output_dir)

    # 遍历输入目录中的所有文件-class 1
    for filename in os.listdir(class_input_dir):
        # 检查文件是否为图像
        if filename.endswith('.jpeg'):
            # 获取输入图像的完整路径
            input_path = os.path.join(class_input_dir, filename)

            # cv2-打开原图并保存到新的路径下
            img = cv2.imread(filename=input_path)
            cv2.imwrite(
                filename=os.path.join(class_output_dir, filename),
                img=img)

            # 数据增强
            # PIL-打开图像
            img = Image.open(fp=input_path)
            img_enhanced = Random_Enhance(img=img)
            # 获取输出图像的完整路径
            output_path = os.path.join(class_output_dir, f'{filename[:-5]}-{suffix}.jpeg')

            # 保存增强后的图像
            img_enhanced.save(output_path)


# 使用示例
if __name__ == "__main__":
    # 获取根目录
    root_path = os.getcwd()
    # 原始数据大小
    num = 5126
    normal_num = 1341
    pneumonia_num = 3875
    # 进行多轮增强
    a = 1
    b = 5
    print(f'总共进行{b - 1}轮数据增强')
    for i in range(a, b):
        print(f'Data enhance turn:', i)
        # 对 train\\NORMAL 里的数据进行增强
        batch_enhance_image(
            class_input_dir=os.path.join(root_path, 'Chest_XRay\\train\\NORMAL'),
            class_output_dir=os.path.join(root_path, 'data_enhance\\train\\NORMAL'),
            suffix=i)
        # 对 train\\PNEUMONIA 里的数据进行增强
        batch_enhance_image(
            class_input_dir=os.path.join(root_path, 'Chest_XRay\\train\\PNEUMONIA'),
            class_output_dir=os.path.join(root_path, 'data_enhance\\train\\PNEUMONIA'),
            suffix=i)
        print('train NORMAL:', normal_num * (i + 1))
        print('train PNEUMONIA', pneumonia_num * (i + 1))
        print('train Total:', num * (i + 1))
        print('-' * 50)
