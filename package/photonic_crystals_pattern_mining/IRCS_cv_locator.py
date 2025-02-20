


def image_preprocessing(path_dir = './dataset', SUB_TRACK_OUTSIDE = True, g_blur_ksize = (5, 5), g_blur_sigmaX = 0, 
                        find_circle_canny_lower_threshold = 5, find_circle_canny_upper_threshold = 15, second_zoom_in_factor = 40, 
                        find_circle_no_circle_canny_lower = 20, find_circle_no_circle_canny_upper = 25, 
                        find_radius_canny_lower_threshold = 5, find_radius_canny_upper_threshold = 15, 
                        r_blur_ksize = (5, 5), r_blur_sigmaX = 0, r_canny_blur = (40, 40)):
    import math
    import pathlib
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import os

    def find_circle_threshold_ver(image_path, TARGET_IMAGE_SIZE):
        # preprocess
        src = cv2.imread(image_path)
        # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # canny = cv2.Canny(gaussian_blur, 5, 15)
        # mask = canny
    
        # use moments to find the center of the circle
        # kernel = np.ones((8, 8), np.uint8)qut
        # mask = cv2.dilate(mask, kernel, iterations=3)
        # M = cv2.moments(mask)
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])

        # obtain bounding box and background removed img
        no_background_img, img_with_bg_fill, circle_mask, bounding_rect = remove_background_and_find_radius(src)
        x, y, w, h = bounding_rect
        circle_diameter = max(w, h)
        print('circle diameter:', circle_diameter)

        gray = cv2.cvtColor(img_with_bg_fill, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray, g_blur_ksize, g_blur_sigmaX)
        canny = cv2.Canny(gaussian_blur, find_circle_canny_lower_threshold, find_circle_canny_upper_threshold)
        canny = cv2.bitwise_and(canny, canny, mask=circle_mask)

        # use bounding box to crop the image
        offset = min(100, x, y)
        offset = offset - 100
        if SUB_TRACK_OUTSIDE:
            # offset = offset - 190
            offset = offset - 100
        cropped_img = src[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_canny = canny[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_no_bg = no_background_img[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_no_bg_fill = img_with_bg_fill[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        if SUB_TRACK_OUTSIDE:
            # future zoom in for rgb_zoom output
            offset = offset - second_zoom_in_factor
        cropped_img_future = src[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]

        # resize the image
        resized_img = cv2.resize(cropped_img, TARGET_IMAGE_SIZE)
        resized_canny = cv2.resize(cropped_canny, TARGET_IMAGE_SIZE)
        resized_no_bg = cv2.resize(cropped_no_bg, TARGET_IMAGE_SIZE)
        resized_no_bg_fill = cv2.resize(cropped_no_bg_fill, TARGET_IMAGE_SIZE)
        resized_cropped_img_future = cv2.resize(cropped_img_future, TARGET_IMAGE_SIZE)

        # generate a fix size circle mask
        circle_mask = np.zeros((TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]), dtype='uint8')
        circle_mask = cv2.circle(circle_mask, (TARGET_IMAGE_SIZE[0] // 2, TARGET_IMAGE_SIZE[1] // 2), TARGET_IMAGE_SIZE[0] // 2, (255, 255, 255), -1)
        no_bg_fix_circle_mask = cv2.bitwise_and(resized_img, resized_img, mask=circle_mask)
        no_bg_fix_circle_mask_canny = cv2.bitwise_and(resized_canny, resized_canny, mask=circle_mask)

        # obtain hist fig
        fig = get_histogram(resized_img)
        no_bg_fig = get_histogram(no_background_img)
        no_bg_bin_fig = get_histogram_bin(no_background_img)

        no_circle_detection_original_img = src.copy()
        no_circle_detection_original_canny_img = cv2.Canny(no_circle_detection_original_img, find_circle_no_circle_canny_lower, find_circle_no_circle_canny_upper)

        return resized_img, resized_canny, fig, resized_no_bg, no_bg_fig, no_bg_bin_fig, resized_no_bg_fill, no_circle_detection_original_img, \
            no_circle_detection_original_canny_img, no_bg_fix_circle_mask, no_bg_fix_circle_mask_canny, resized_cropped_img_future

    def mouse_call_back(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('x:', x, 'y:', y)
            param[0] = param[0] + 1
            param[1] = x
            param[2] = y

    def remove_background_and_find_radius(img, fill_with_mean_color=False):
        # preprocess
        hh, ww = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray, r_blur_ksize, r_blur_sigmaX)
        canny = cv2.Canny(gaussian_blur, find_radius_canny_lower_threshold, find_radius_canny_upper_threshold)
        blur = cv2.blur(canny, r_canny_blur)

        # get the (largest) contour
        contours = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        bounding_rect = cv2.boundingRect(big_contour)

        # generate the mask
        mask = np.zeros((hh, ww), dtype='uint8')
        cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), cv2.FILLED)

        # ask user to click on circle center
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        i = [0, 0, 0]
        cv2.setMouseCallback('win', mouse_call_back, i)
        # cv2.imshow('win', img)
        # while i[0] < 1:
        #     cv2.waitKey(1)

        # check if a mask is circle shape
        # M = cv2.moments(mask)
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        cx = i[1]
        cy = i[2]
        radius = [math.sqrt((contour_point[0][0] - cx)**2 + (contour_point[0][1] - cy)**2) for contour_point in big_contour]
        mean_radius = np.mean(radius)
        std_radius = np.std(radius)

        # clean contour point based on gaussian distribution
        clean_contour = np.array([contour_point for contour_point in big_contour if math.sqrt((contour_point[0][0] - cx)**2 + (contour_point[0][1] - cy)**2) - mean_radius < std_radius])

        # create new mask based on clean contour
        clean_mask = np.zeros((hh, ww), dtype='uint8')
        cv2.drawContours(clean_mask, [clean_contour], 0, (255, 255, 255), cv2.FILLED)

        # recalculate the moments and radius
        M = cv2.moments(clean_mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        new_radius = [math.sqrt((contour_point[0][0] - cx) ** 2 + (contour_point[0][1] - cy) ** 2) for contour_point in clean_contour]
        new_mean_radius = np.mean(new_radius)
        new_std_radius = np.std(new_radius)

        # create circle mask
        circle_mask = np.zeros((hh, ww), dtype='uint8')
        circle_mask = cv2.circle(circle_mask, (cx, cy), int(new_mean_radius), (255, 255, 255), -1)
        # circle_mask_result = cv2.bitwise_and(img, img, mask=circle_mask)
        # cv2.imshow('circle_mask_result', circle_mask_result)
        # cv2.waitKey(0)

        mean_color = cv2.mean(img)
        result_with_bg_fill = cv2.bitwise_and(img, img, mask=circle_mask)
        result_with_bg_fill[np.where((result_with_bg_fill == [0, 0, 0]).all(axis=2))] = [mean_color[0], mean_color[1], mean_color[2]]

        result = cv2.bitwise_and(img, img, mask=circle_mask)
        # cv2.imshow('result', result)
        # cv2.waitKey()

        # get the (largest) contour
        contours = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        bounding_rect = cv2.boundingRect(big_contour)

        #generate return mask
        circle_mask = np.zeros((hh, ww), dtype='uint8')
        circle_mask = cv2.circle(circle_mask, (cx, cy), int(new_mean_radius)-5, (255, 255, 255), -1)
        return result, result_with_bg_fill, circle_mask, bounding_rect


    def get_histogram(raw_img):
        # get the histogram of the image
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        color = ('r', 'g', 'b')
        image_hist_vector = np.array([])
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [1, 256])
            image_hist_vector = np.append(image_hist_vector, histr)
            ax.plot(histr, color=col)
            ax.set_xlim([1, 256])
        return ax

    def get_histogram_bin(raw_img):
        # get the histogram of the image
        HIST_BINS = 8
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        color = ('r', 'g', 'b')
        image_hist_vector = np.array([])
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, col in enumerate(color):
            np_his, _ = np.histogram(img[:, :, i], bins=HIST_BINS, range=(1, 255))
            image_hist_vector = np.append(image_hist_vector, np_his)
            ax.plot(np_his, color=col)
            ax.set_xlim([0, HIST_BINS-1])
        return ax





    print('cv_locator start')
    TARGET_IMAGE_SIZE = (1000, 1000)
    data_dir = pathlib.Path(path_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(str(image_count) + 'images found')
    for output_path in ['./output/rgb', './output/rgb_no_circle_det', './output/canny', './output/canny_no_circle_det', 
                 './output/no_bg', './output/no_bg_fill', './output/no_bg_fix_circle_mask', 
                 './output/no_bg_fix_circle_mask_canny', './output/rgb_zoom', './output/hist', './output/no_bg_hist', './output/no_bg_bin_hist']:
        os.makedirs(output_path, exist_ok=True)
    for path in data_dir.glob('*/*.jpg'):
        print('processing ' + str(path))
        img = cv2.imread(str(path))
        image_name = path.name
        # if image_name == 'M-DMMP-DMMP-10-3M-1.jpg':
        #     print('Bad Image Detected, wont process')
        #     continue
        if img.shape[0] < 500:
            continue
        resized_img, bounding_rect, ax, no_background_img, no_bg_ax, no_bg_bin_ax, img_with_bg_fill, \
        no_circle_detection_original_img, no_circle_detection_original_canny_img, no_bg_fix_circle_mask, \
        no_bg_fix_circle_mask_canny, resized_cropped_img_future = find_circle_threshold_ver(str(path), TARGET_IMAGE_SIZE)
        # try:
        #     resized_img, bounding_rect, ax, no_background_img, no_bg_ax, no_bg_bin_ax, img_with_bg_fill, \
        #     no_circle_detection_original_img, no_circle_detection_original_canny_img = find_circle_threshold_ver(str(path), TARGET_IMAGE_SIZE)
        # except Exception as e:
        #     print(e)
        #     print('error found when processing ' + str(path))
        #     cv2.imshow('img', img)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        #     continue
        print(0)
        cv2.imwrite('./output/rgb/{0}'.format(image_name), resized_img)
        print(1)
        cv2.imwrite('./output/rgb_no_circle_det/{0}'.format(image_name), no_circle_detection_original_img)
        print(2)
        cv2.imwrite('./output/canny/{0}'.format(image_name), bounding_rect)
        print(3)
        cv2.imwrite('./output/canny_no_circle_det/{0}'.format(image_name), no_circle_detection_original_canny_img)
        print(4)
        cv2.imwrite('./output/no_bg/{0}'.format(image_name), no_background_img)
        print(5)
        cv2.imwrite('./output/no_bg_fill/{0}'.format(image_name), img_with_bg_fill)
        print(6)
        cv2.imwrite('./output/no_bg_fix_circle_mask/{0}'.format(image_name), no_bg_fix_circle_mask)
        print(7)
        cv2.imwrite('./output/no_bg_fix_circle_mask_canny/{0}'.format(image_name), no_bg_fix_circle_mask_canny)
        print(8)
        cv2.imwrite('./output/rgb_zoom/{0}'.format(image_name),resized_cropped_img_future)
        print(9)
        ax.figure.savefig('./output/hist/{0}'.format(image_name))
        print(10)
        no_bg_ax.figure.savefig('./output/no_bg_hist/{0}'.format(image_name))
        print(11)
        no_bg_bin_ax.figure.savefig('./output/no_bg_bin_hist/{0}'.format(image_name))
        print(12)

    # path_dict = './P6-DeviceExamples-MAP-20211216'
    # images_list = [f for f in listdir(path_dict) if isfile(join(path_dict, f))]
    # result = []
    # for images_name in images_list:
    #     if images_name.startswith('.'):
    #         continue
    #     image = find_circle_threshold_ver(path_dict + '/{0}'.format(images_name))
    #     # cv2.imwrite('./P6-DeviceExamples-MAP-20211216/output/{0}'.format(images_name), image)
    #     cv2.imwrite('./P6-DeviceExamples-MAP-20211229/output/{0}'.format(images_name), image)

    # test
    # test_image_path = './dataset\M-DMMP-KF6P-10-6M\M-DMMP-KF6P-10-6M-1.jpg'
    # test_image_path = './dataset\M-DMMP-KF6P-10-6M\M-DMMP-KF6P-10-6M-2.jpg'
    # find_circle_threshold_ver(test_image_path, (1000, 1000))
