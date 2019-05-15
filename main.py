import imageio
import numpy as np
import matplotlib.pyplot as plt

compartments_quantity = 50


def normalize(image):
    """
    This method normalize the image value
    :param image: value after the conversion to grayscale
    :return: normalized output
    """
    offset = 255.0 / compartments_quantity
    output = np.zeros((image.shape[0], image.shape[0]))
    i = 0
    tmp = 0
    while i < 255.0:
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                if i < image[x][y] <= i + offset:
                    output[x][y] = tmp
        tmp = tmp + 1
        i = i + offset
    return output


def calculate_sharpness(coincident_matrix):
    """
    Method calculate the sharpness
    as the sum of diagonal pixels
    :param coincident_matrix: concident matrix
    :return: number of pixels on the diagonal
    """
    sum_diag_pixels = 0
    for i in range(0, coincident_matrix.shape[0]):
        for j in range(0, coincident_matrix.shape[1]):
            if coincident_matrix[i][j] != 0:
                sum_diag_pixels += 1
    return sum_diag_pixels


def convert_rgb_2_gray(img_rgb):
    """
    Method convert image from rgb to grayscale
    :param img_rgb: image matrix in rgb
    :return: image value in grayscale
    """
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    img_gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return img_gray


def create_coincident_matrix(img, dlx, dly):
    """
    Method create the coincident_matrix
    from image in grayscale.
    :param img: Image in grayscale
    :param dlx: lenght of x
    :param dly: length of y
    :return: Coincident matrix
    """
    p_matrix = np.max(img).astype(int) + 1
    coincident_matrix = np.zeros((p_matrix, p_matrix))
    height = len(img)
    width = len(img[0])

    for x in range(0, height - dlx):
        for y in range(0, width - dly):
            xx = img[x, y]
            xx = xx.astype(int)
            yy = img[x + dlx, y + dly]
            yy = yy.astype(int)
            coincident_matrix[xx, yy] = coincident_matrix[xx, yy] + 1

    return coincident_matrix


if __name__ == '__main__':

    # First image

    img_1 = imageio.imread('oko1.png')  # Loading file to matrix
    img_1 = convert_rgb_2_gray(img_1)   # Converting rgb -> grayscale
    img_1 = normalize(img_1)            # Value normalization

    plt.subplot(2, 2, 1)                # Cut the plot workspace to nrows x ncols x index
    # Draw image
    plt.imshow(img_1, cmap=plt.cm.gray, vmin=0, vmax=compartments_quantity)
    plt.title("oko1.png")               # Show the plot title

    plt.subplot(2, 2, 2)
    # Draw histogram
    plt.imshow(create_coincident_matrix(img_1, 0, 1), cmap=plt.cm.gray, vmin=0, vmax=compartments_quantity)
    plt.title("macierz wspowwystepowania oko1.png")

    print("Ostrosc oko1: " + str(calculate_sharpness(create_coincident_matrix(img_1, 0, 1))))

    # Second image

    img_2 = imageio.imread('oko2.png')
    img_2 = convert_rgb_2_gray(img_2)
    img_2 = normalize(img_2)

    plt.subplot(2, 2, 3)
    plt.imshow(img_2, cmap=plt.cm.gray, vmin=0, vmax=compartments_quantity)
    plt.title("oko2.png")

    plt.subplot(2, 2, 4)
    plt.imshow(create_coincident_matrix(img_2, 0, 1), cmap=plt.cm.gray, vmin=0, vmax=compartments_quantity)
    plt.title("macierz wspowwystepowania oko2.png")

    print("Ostrosc oko2: " + str(calculate_sharpness(create_coincident_matrix(img_2, 0, 1))))

    # Show all calculations and histograms

    plt.show()
