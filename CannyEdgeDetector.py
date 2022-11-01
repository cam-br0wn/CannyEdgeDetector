import numpy as np
from PIL import Image, ImageFilter


def convert_grayscale(img_arr):
    gray_arr = []
    for row in range(len(img_arr)):
        gray_arr.append([])
        for col in range(len(img_arr[row])):
            color_sum = 0
            for color in range(len(img_arr[row][col])):
                color_sum += img_arr[row][col][color]
            gray_arr[row].append(int(color_sum / len(img_arr[row][col])))
    return gray_arr


# NOT IN A WORKING STATE, ROTATES AND DOESN'T BLUR PROPERLY
def gaussian_smooth(img_arr, n, sigma):
    img_arr = np.array(convert_grayscale(img_arr))
    gaussian_arr = np.zeros((len(img_arr), len(img_arr[0])))
    kernel = np.zeros((n, n))
    kernel_indices = np.zeros((n, n, 2))
    for row in range(n):
        for col in range(n):
            kernel[row, col] = 1 / (2 * np.pi * np.power(sigma, 2)) * np.exp((-1 * ((col - int(n/2))**2 + (row - int(n/2))**2) / (2 * (sigma**2))))
            kernel_indices[row, col] = [col - int(n/2), row - int(n/2)]
    for row in range(len(img_arr)):
        for col in range(len(img_arr[0])):
            new_val = 0
            val_ct = 1
            for ker_row in range(n):
                for ker_col in range(n):
                    col_idx = int(col + kernel_indices[ker_row, ker_col, 0])
                    row_idx = int(row + kernel_indices[ker_row, ker_col, 1])
                    if (col_idx > -1) and (row_idx > -1) and (row_idx < img_arr.shape[1]) and (col_idx < img_arr.shape[0]):
                        val_ct += 1
                        new_val += kernel[ker_row, ker_col] * img_arr[col_idx, row_idx]
            new_val = int(new_val/val_ct)
            gaussian_arr[row, col] = new_val
    return gaussian_arr


def image_gradient(gauss_arr):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    transformed = np.zeros(gauss_arr.shape)
    for row in range(gauss_arr.shape[0]):
        for col in range(gauss_arr.shape[1]):
            new_x = 0
            new_y = 0
            for sob_r in range(3):
                for sob_c in range(3):
                    r_idx = row - 1 + sob_r
                    c_idx = col - 1 + sob_c
                    if (r_idx > -1) and (r_idx < gauss_arr.shape[0]) and (c_idx > -1) and (c_idx < gauss_arr.shape[1]):
                        new_x += sobel_x[sob_r, sob_c] * gauss_arr[r_idx, c_idx]
                        new_y += sobel_y[sob_r, sob_c] * gauss_arr[r_idx, c_idx]
            transformed[row, col] = min(255, int(np.sqrt(new_x**2 + new_y**2)))
    return transformed


# helper function for finding thresholds
def gen_freq_dict(arr):
    freq_dict = {}
    for item in arr:
        if int(item) in freq_dict.keys():
            freq_dict[int(item)] += 1
        else:
            freq_dict[int(item)] = 0
    return freq_dict


# helper function for finding thresholds
def gen_prob_dict(dictionary, n):
    prob_dict = {}
    for key in dictionary.keys():
        prob_dict[key] = dictionary[key]/n
    return prob_dict


# helper function for finding thresholds
def cdf_from_prob_dict(prob_dict, i):
    cumulative_prob = 0
    counter = 0
    while counter <= i:
        if counter in prob_dict.keys():
            cumulative_prob += prob_dict[counter]
        counter += 1
    return cumulative_prob


def determine_thresholds(grad_arr):
    # here we want to determine the high threshold by taking the histogram and setting a cutoff point
    # low threshold will then be half the high threshold
    high_thresh = 0
    # first, collapse the gradient array into one dimension
    flat_grad = grad_arr.flatten()
    # now we want to calculate the non-edge proportion
    # this value should be tweaked as necessary (anything less than it is deemed a non-edge pixel)
    non_edge_thresh = 128
    non_edge_ct = 0
    for pix in flat_grad:
        if pix < non_edge_thresh:
            non_edge_ct += 1
    # we will use this percentage as the cutoff for our cdf
    non_edge_pct = non_edge_ct / len(flat_grad)
    freq_dict = gen_freq_dict(flat_grad)
    prob_dict = dict(sorted(gen_prob_dict(freq_dict, len(flat_grad)).items()))
    for key in prob_dict.keys():
        if cdf_from_prob_dict(prob_dict, key) > non_edge_pct:
            high_thresh = key
            break
    low_thresh = int(high_thresh / 2)
    return high_thresh, low_thresh


def supress_non_maxima(grad_arr, high_thresh, low_thresh):
    print('not implemented yet for reason: yolo swag 300')
    return


def main():
    image_list = ['gun1.bmp', 'joy1.bmp', 'lena.bmp', 'pointer1.bmp', 'test1.bmp']

    # DEBUG ITEMS
    test_grayscales = False
    test_gaussian = True
    test_gradient = False
    bypass_gaussian = True
    test_thresh = False

    # test grayscale capability
    if test_grayscales:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            image_bmp = np.array(image_bmp)
            grayscale_image = convert_grayscale(image_bmp)
            converted_image = Image.fromarray(np.array(grayscale_image).astype(np.uint8))
            converted_image.save('grayscales/' + image_file)

    if test_gaussian:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            if not bypass_gaussian:
                image_bmp = np.array(image_bmp)
                gaussian_image = gaussian_smooth(image_bmp, 3, 0.1)
            else:
                image_bmp = image_bmp.filter(ImageFilter.GaussianBlur)
                image_bmp = np.array(image_bmp)
                gaussian_image = np.array(convert_grayscale(image_bmp))
            converted_image = Image.fromarray(gaussian_image.astype(np.uint8))
            converted_image.save('gaussian/' + image_file)

    if test_gradient:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            if not bypass_gaussian:
                image_bmp = np.array(image_bmp)
                gradient_image = image_gradient(gaussian_smooth(image_bmp, 3, 0.1))
            else:
                image_bmp = image_bmp.filter(ImageFilter.GaussianBlur)
                image_bmp = np.array(image_bmp)
                image_bmp = np.array(convert_grayscale(image_bmp))
                gradient_image = image_gradient(image_bmp)
            converted_image = Image.fromarray(gradient_image.astype(np.uint8))
            converted_image.save('gradient/' + image_file)

    if test_thresh:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            if not bypass_gaussian:
                image_bmp = np.array(image_bmp)
                gradient_image = image_gradient(gaussian_smooth(image_bmp, 3, 0.1))
            else:
                image_bmp = image_bmp.filter(ImageFilter.GaussianBlur)
                image_bmp = np.array(image_bmp)
                image_bmp = np.array(convert_grayscale(image_bmp))
                gradient_image = image_gradient(image_bmp)
            print(determine_thresholds(gradient_image))
    return


if __name__ == '__main__':
    main()
