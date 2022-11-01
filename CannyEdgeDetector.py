import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def convolute(x, y):
    padding_amt = y.shape[0] // 2
    padded = np.zeros((x.shape[0] + (2 * padding_amt), x.shape[1] + (2 * padding_amt)))
    padded[padding_amt:x.shape[0] + padding_amt, padding_amt:x.shape[1] + padding_amt] = x
    result = np.zeros(padded.shape)
    for row in range(padding_amt, padding_amt + x.shape[0]):
        for col in range(padding_amt, padding_amt + x.shape[1]):
            sum = 0
            for y_row in range(y.shape[0]):
                for y_col in range(y.shape[1]):
                    sum += y[y_row, y_col] * padded[row - padding_amt + y_row, col - padding_amt + y_col]
            result[row, col] = sum
    return result[padding_amt:x.shape[0] + padding_amt, padding_amt:x.shape[1] + padding_amt]


def convert_grayscale(img_arr):
    gray_arr = []
    for row in range(len(img_arr)):
        gray_arr.append([])
        for col in range(len(img_arr[row])):
            # color_sum = 0
            # for color in range(len(img_arr[row][col])):
            #     color_sum += img_arr[row][col][color]
            # gray_arr[row].append(int(color_sum / len(img_arr[row][col])))
            pix = img_arr[row][col]
            gray_arr[row].append(int((0.299*pix[0]) + (0.587 * pix[1]) + (0.114 * pix[2])))
    return gray_arr


def gaussian_smooth(img_arr, n, sigma):
    img_arr = np.array(convert_grayscale(img_arr))
    kernel = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            kernel[row, col] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(
                (-1 * ((col - (n//2))**2 + (row - (n//2))**2)) / (2 * (sigma ** 2)))
    return convolute(img_arr, kernel)


def image_gradient(gauss_arr):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    transformed_x = convolute(gauss_arr, sobel_x)
    transformed_y = convolute(gauss_arr, sobel_y)
    magnitude = np.zeros(gauss_arr.shape)
    direction = np.zeros(gauss_arr.shape)
    for row in range(magnitude.shape[0]):
        for col in range(magnitude.shape[1]):
            magnitude[row, col] = min([255, np.sqrt((transformed_x[row, col]**2)+(transformed_y[row,col]**2))])
            direction[row, col] = np.arctan2(transformed_y[row, col], transformed_x[row, col])
    return magnitude, direction


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
        prob_dict[key] = dictionary[key] / n
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


def stretch_hist(img_arr, img_1d):
    stretched_img = []
    # frequency of each grayscale value
    freq_dict = gen_freq_dict(img_1d)
    # number of pixels
    n = len(img_arr) * len(img_arr[0])
    # probability of each grayscale value
    prob_dict = dict(sorted(gen_prob_dict(freq_dict, n).items()))
    for row in range(len(img_arr)):
        stretched_img.append([])
        for col in range(len(img_arr[row])):
            stretched_img[row].append(int(255 * cdf_from_prob_dict(prob_dict, img_arr[row][col])))
    return np.array(stretched_img)


def determine_thresholds(grad_arr):
    # here we want to determine the high threshold by taking the histogram and setting a cutoff point
    # low threshold will then be half the high threshold
    high_thresh = 0
    # first, collapse the gradient array into one dimension
    flat_grad = grad_arr.flatten()
    stretched_grad_arr = stretch_hist(grad_arr, flat_grad)
    flat_stretch_grad_arr = stretched_grad_arr.flatten()
    # plt.hist(flat_grad, bins=255)
    # plt.hist(flat_stretch_grad_arr, bins=255)
    # plt.show()
    # now we want to calculate the non-edge proportion
    # this value should be tweaked as necessary (anything less than it is deemed a non-edge pixel)
    non_edge_thresh = 156
    non_edge_ct = 0
    for pix in flat_stretch_grad_arr:
        if pix < non_edge_thresh:
            non_edge_ct += 1
    # we will use this percentage as the cutoff for our cdf
    non_edge_pct = non_edge_ct / len(flat_stretch_grad_arr)
    freq_dict = gen_freq_dict(flat_stretch_grad_arr)
    prob_dict = dict(sorted(gen_prob_dict(freq_dict, len(flat_stretch_grad_arr)).items()))
    for key in prob_dict.keys():
        if cdf_from_prob_dict(prob_dict, key) > non_edge_pct:
            high_thresh = key
            break
    low_thresh = int(high_thresh / 2)
    return high_thresh, low_thresh


def suppress_non_maxima(mag_arr, dir_arr, high_thresh, low_thresh):
    print('not implemented yet for reason: yolo swag 300')
    return


def main():
    image_list = ['gun1.bmp', 'joy1.bmp', 'lena.bmp', 'pointer1.bmp', 'test1.bmp']

    # DEBUG ITEMS
    test_grayscales = 1
    test_gaussian = 0
    test_gradient = 1
    test_thresh = 0
    n = 5
    sigma = np.sqrt(n)

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
            image_bmp = np.array(image_bmp)
            gaussian_image = gaussian_smooth(image_bmp, n, sigma)
            converted_image = Image.fromarray(gaussian_image.astype(np.uint8))
            converted_image.save('gaussian/' + image_file)

    if test_gradient:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            image_bmp = np.array(image_bmp)
            gradient_image = image_gradient(gaussian_smooth(image_bmp, n, sigma))[0]
            converted_image = Image.fromarray(gradient_image.astype(np.uint8))
            converted_image.save('gradient/' + image_file)

    if test_thresh:
        for image_file in image_list:
            image_bmp = Image.open(image_file)
            image_bmp = np.array(image_bmp)
            gradient_image = image_gradient(gaussian_smooth(image_bmp, n, sigma))[0]
            print(determine_thresholds(gradient_image))

    return


if __name__ == '__main__':
    main()
