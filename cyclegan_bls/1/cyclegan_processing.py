# from .cyclegan_noise_removal import _split_img, _merge_imgs
import cv2
import numpy as np
# from tensorflow.keras.preprocessing import image


def split_img(img, res):
    """Split an image into smaller patches of given shape

    Args:
        img (<numpy.array>): Image as numpy array
        split_res (tuple): Resolution of each splits

    Returns:
        list: list of splitted images
    """

    # result_img = []
    image_list = np.zeros((1, 256, 512, 1))
    split_height = res[0]
    split_width = res[1]

    # img = cv2.imread(img_loc, cv2.IMREAD_COLOR)

    source_height = img.shape[0]
    source_width = img.shape[1]

    if (source_height < split_height) or (source_width < split_width):
        print("Input image dimension is less than " + str(split_height) + " x " + str(split_width))

    ht = 0
    while ht < source_height:
        wd = 0
        while wd < source_width:
            tmp_img = img

            if ht + split_height > source_height:
                diff = (ht + split_height) - source_height
                ht -= diff

            if wd + split_width > source_width:
                diff = (wd + split_width) - source_width
                wd -= diff

            img = img[ht:ht + split_height, wd:wd + split_width]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = image.img_to_array(img).astype('float32')
            img = np.expand_dims(img, 0)

            if ht == 0 and wd == 0:
                image_list = np.expand_dims(img, axis=0)
            else:
                img = np.expand_dims(img, axis=0)
                image_list = np.concatenate((image_list, img), axis=0)

            img = tmp_img
            wd += split_width

        ht += split_height

    return image_list


def merge_imgs(imgs, res):
    """Given a shape merge a list of image patches to get a whole image

    Args:
        img (<numpy.array>): Image as numpy array
        res (tuple): Resolution of the whole image

    Returns:
        <numpy.array>: Image as numpy array
    """

    target_height = res[0]
    target_width = res[1]

    result_img = []
    i = 0

    while len(result_img) == 0 or result_img.shape[0] < target_height:

        tmp_img = []

        while len(tmp_img) == 0 or tmp_img.shape[1] < target_width:

            if len(tmp_img) == 0:
                tmp_img = imgs[i]
            else:
                if tmp_img.shape[1] + imgs[i].shape[1] > target_width:
                    extra_width = tmp_img.shape[1] + imgs[i].shape[1] - target_width
                    cropped_img = imgs[i][:, extra_width:]
                    tmp_img = np.concatenate((tmp_img, cropped_img), axis=1)
                else:
                    tmp_img = np.concatenate((tmp_img, imgs[i]), axis=1)
            i += 1

        if len(result_img) == 0:
            result_img = tmp_img
        else:
            if result_img.shape[0] + tmp_img.shape[0] > target_height:
                extra_height = result_img.shape[0] + tmp_img.shape[0] - target_height
                cropped_img = tmp_img[extra_height:, :]
                result_img = np.concatenate((result_img, cropped_img), axis=0)
            else:
#                 local_time = time.time()
                result_img = np.concatenate((result_img, tmp_img), axis=0)
#                 print(time.time() - local_time)


    return result_img




# img = cv2.imread('/home/sazzad/Documents/GitHub/custom_bls/typewriter.jpg')
# print('original image shape', img.shape)

# split_images = split_img(img, [256, 512])

# print(len(split_images))
# print(split_images[0].shape)

# ############# run cyclegan here ################

# # recons_img = _merge_imgs(clean_images, img.shape)


# recons_img = merge_imgs(split_images, img.shape)
# recons_img = cv2.cvtColor(recons_img,cv2.COLOR_GRAY2RGB)


# print('recons_img shape', recons_img.shape)