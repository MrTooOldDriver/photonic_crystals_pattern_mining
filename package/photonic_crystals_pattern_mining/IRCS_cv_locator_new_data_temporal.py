# import glob
# import math
# import os
# import re
# import pathlib
# import cv2

# # 用于检测晶球位置和做对应剪裁
# import numpy as np
# from matplotlib import pyplot as plt

def image_preprocessing(path_dir = './dataset', SUB_TRACK_OUTSIDE = True, ADJUST_WITH_CIRCLE_RADIUS = True, g_blur_ksize = (5, 5), g_blur_sigmaX = 0, 
                        find_circle_canny_lower_threshold = 5, find_circle_canny_upper_threshold = 15, second_zoom_in_factor = 40, 
                        find_circle_no_circle_canny_lower = 20, find_circle_no_circle_canny_upper = 25, 
                        find_radius_canny_lower_threshold = 5, find_radius_canny_upper_threshold = 15):
    import glob
    import math
    import os
    import re
    import pathlib
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    def find_circle_threshold_ver(image_path, TARGET_IMAGE_SIZE, radius_limit):
        # preprocess
        src = cv2.imread(image_path)
        src = cv2.copyMakeBorder(src, top=100, bottom=100, left=100, right=100, borderType=cv2.BORDER_WRAP)
        # crop image into a square
        # src = src[:, 128:1152]

        # obtain bounding box and background removed img
        no_background_img, img_with_bg_fill, circle_mask, bounding_rect = remove_background_and_find_radius(src)
        x, y, w, h = bounding_rect
        circle_diameter = max(w, h)
        print('circle diameter:', circle_diameter)

        gray = cv2.cvtColor(img_with_bg_fill, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray, g_blur_ksize, g_blur_sigmaX)
        canny = cv2.Canny(gaussian_blur, find_circle_canny_lower_threshold, find_circle_canny_upper_threshold)
        canny = cv2.bitwise_and(canny, canny, mask=circle_mask)

        if not ADJUST_WITH_CIRCLE_RADIUS:
            target_diameter = radius_limit
            if target_diameter > circle_diameter:
                offset = (target_diameter - circle_diameter) // 2
        else:
            offset = 0
        if SUB_TRACK_OUTSIDE:   # TODO still second zoomin factor?
            offset = offset - 120
        # Add cropping with padding
        cropped_img = src[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_canny = canny[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_no_bg = no_background_img[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
        cropped_no_bg_fill = img_with_bg_fill[y - offset:y + circle_diameter + offset, x - offset:x + circle_diameter + offset]
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
        # cv2.imshow('gray', gray)
        # gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(gray, 15, 50) # TODO check if needed to parameterize

        # add pixel value smaller than 48,48,48 into canny
        for i in range(hh):
            for j in range(ww):
                if gray[i][j] < 70:
                    canny[i][j] = 255

        blur = cv2.blur(canny, (3, 3))
        # cv2.imshow('blur', blur)

        # get the (largest) contour
        contours = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        # bounding_rect = cv2.boundingRect(big_contour)

        # generate the mask
        mask = np.zeros((hh, ww), dtype='uint8')
        cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), cv2.FILLED)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        # ask user to click on circle center
        # cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        # i = [0, 0, 0]
        # cv2.setMouseCallback('win', mouse_call_back, i)
        # cv2.imshow('win', img)
        # while i[0] < 1:
        #     cv2.waitKey(1)

        # check if a mask is circle shape
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # cx = i[1]
        # cy = i[2]
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


    if __name__ == '__main__':
        print('cv_locator start')
        TARGET_IMAGE_SIZE = (1000, 1000)
        path_dir = './new_data_temporal_20231218'
        data_dir = pathlib.Path(path_dir)
        image_count = len(list(data_dir.glob('*/*/*/*.jpg')))
        print(str(image_count) + 'images found')
        i = 0

        if ADJUST_WITH_CIRCLE_RADIUS:
            output_folder_name = 'output_new_data_temporal'
        else:
            output_folder_name = 'output_new_data_temporal_no_radius_adjust'

        # create output folder
        pathlib.Path('./' + output_folder_name).mkdir(parents=True, exist_ok=True)
        pathlib.Path('./' + output_folder_name + '/rgb').mkdir(parents=True, exist_ok=True)

        # pathlib.Path('./output_new_data_temporal/rgb_no_circle_det').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/canny').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/canny_no_circle_det').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg_fill').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg_fix_circle_mask').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg_fix_circle_mask_canny').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/rgb_zoom').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/hist').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg_hist').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('./output_new_data_temporal/no_bg_bin_hist').mkdir(parents=True, exist_ok=True)

        identifier = ''

        image_limit = {
            'C4': 12,
            'C6': 18,
            'C8': 16
        }

        radius_limit = {
            'C4': 890,
            'C6': 920,
            'C8': 850
        }


        for path in sorted(data_dir.glob('*/*/*')):

            # Get list of all files
            files = list(glob.glob((str(path) + '\*.jpg')))

            # Sort files based on numeric names
            files.sort(key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[0]))

            for file_path in files:
                # file_path = './new_data_temporal_20231218\C6\M-BF4/1/1.jpg'
                print('processing ' + str(file_path))
                file_path = pathlib.Path(file_path)
                img = cv2.imread(str(file_path))

                solution_name = file_path.parent.parent.parent.name
                ion_name = file_path.parent.parent.name
                sequence_name = file_path.parent.name

                current_identifier = solution_name + '-' + ion_name + '-' + sequence_name
                if current_identifier != identifier:
                    print('new identifier: ' + current_identifier)
                    print('previous identifier %s has %d images' % (identifier, i))
                    identifier = current_identifier
                    i = 0

                image_name = solution_name + '-' + ion_name + '-' + sequence_name + '-' + str(i) + '.jpg'

                if i >= image_limit[solution_name]:
                    print('skip image %s' % (image_name))
                    continue

                resized_img, bounding_rect, ax, no_background_img, no_bg_ax, no_bg_bin_ax, img_with_bg_fill, \
                no_circle_detection_original_img, no_circle_detection_original_canny_img, no_bg_fix_circle_mask, \
                no_bg_fix_circle_mask_canny, resized_cropped_img_future = find_circle_threshold_ver(str(file_path), TARGET_IMAGE_SIZE, radius_limit[solution_name])

                cv2.imwrite('./' + output_folder_name + '/rgb/{0}'.format(image_name), resized_img)
                # cv2.imwrite('./output_new_data_temporal/rgb_no_circle_det/{0}'.format(image_name), no_circle_detection_original_img)
                # cv2.imwrite('./output_new_data_temporal/canny/{0}'.format(image_name), bounding_rect)
                # cv2.imwrite('./output_new_data_temporal/canny_no_circle_det/{0}'.format(image_name), no_circle_detection_original_canny_img)
                # cv2.imwrite('./output_new_data_temporal/no_bg/{0}'.format(image_name), no_background_img)
                # cv2.imwrite('./output_new_data_temporal/no_bg_fill/{0}'.format(image_name), img_with_bg_fill)
                # cv2.imwrite('./output_new_data_temporal/no_bg_fix_circle_mask/{0}'.format(image_name), no_bg_fix_circle_mask)
                # cv2.imwrite('./output_new_data_temporal/no_bg_fix_circle_mask_canny/{0}'.format(image_name), no_bg_fix_circle_mask_canny)
                # cv2.imwrite('./output_new_data_temporal/rgb_zoom/{0}'.format(image_name),resized_cropped_img_future)
                # ax.figure.savefig('./output_new_data_temporal/hist/{0}'.format(image_name))
                # no_bg_ax.figure.savefig('./output_new_data_temporal/no_bg_hist/{0}'.format(image_name))
                # no_bg_bin_ax.figure.savefig('./output_new_data_temporal/no_bg_bin_hist/{0}'.format(image_name))

                plt.close('all')

                i += 1
