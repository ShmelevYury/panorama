import cv2
import numpy as np
from tqdm import tqdm
import numba


class Panorama:
    def __init__(self, file_name, shift=10):
        self.file_name = file_name
        self.shift = shift

    @staticmethod
    def _FindFeatures(frame):
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        desc = cv2.xfeatures2d.SIFT_create()
        (keypoints, features) = desc.detectAndCompute(gray_img, None)
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        return keypoints, features

    @staticmethod
    def getH(points_from, points_to):
        A = []
        for i in range(len(points_from)):
            x, y = points_from[i]
            x_hat, y_hat = points_to[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_hat, y * x_hat, x_hat])
            A.append([0, 0, 0, -x, -y, -1, x * y_hat, y * y_hat, y_hat])

        A = np.asarray(A)

        _, sigma, V = np.linalg.svd(A, full_matrices=True)

        return (V[-1]).reshape(3, 3)

    @staticmethod
    @numba.njit
    def norm(left_pts, right_pts):
        diff_vec = (left_pts - right_pts) ** 2
        norm = np.sqrt(diff_vec[:, 0] + diff_vec[:, 1])
        return norm

    @staticmethod
    def findHomography(ptsL, ptsR, thresh, use_mod=True):
        p = 0.99
        num_of_points = ptsL.shape[0]
        s = 4
        iter_num = 0
        indexes = np.arange(num_of_points)
        inlier_ratio = 0.73
        num_of_iterations = np.log(1 - p) / np.log(1 - (1 - inlier_ratio) ** s)

        all_scores = []
        all_points = []
        while iter_num < num_of_iterations:
            sample_points = np.random.choice(indexes, size=s)
            H = panorama.getH(ptsL[sample_points], ptsR[sample_points])
            transformedL = cv2.perspectiveTransform(np.array([ptsL]), H)

            num_of_inliers = (Panorama.norm(transformedL[0], ptsR) < thresh).sum()

            all_scores.append(num_of_inliers)
            all_points.append(sample_points)
            iter_num += 1

        all_scores = np.asarray(all_scores)
        all_points = np.asarray(all_points)

        ind = all_scores.argmax()
        cur_inliers = all_scores[ind]
        prev_inliers = None
        H_4 = panorama.getH(ptsL[all_points[ind]], ptsR[all_points[ind]])

        while True:
            transformed_4 = cv2.perspectiveTransform(np.array([ptsL]), H_4)
            inliers_mask = (Panorama.norm(transformed_4[0], ptsR) < thresh)

            H_all = panorama.getH(ptsL[inliers_mask], ptsR[inliers_mask])

            transformed_all = cv2.perspectiveTransform(np.array([ptsL[inliers_mask]]), H_all)
            prev_inliers = cur_inliers
            cur_inliers = (Panorama.norm(transformed_all[0], ptsR[inliers_mask]) < thresh).sum()

            if cur_inliers <= prev_inliers:
                return H_4
            elif not use_mod:
                return H_all
            H_4 = H_all

    @staticmethod
    def _FindTransform(keypointsL, keypointsR, featuresL, featuresR, use_custom=False, use_mod=True):
        ratio = 0.85
        thresh = 5
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresL, featuresR, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        ptsL = np.float32([keypointsL[i] for (_, i) in matches])
        ptsR = np.float32([keypointsR[i] for (i, _) in matches])

        if use_custom:
            H = Panorama.findHomography(ptsL, ptsR, thresh, use_mod=use_mod)
            status = None
        else:
            (H, status) = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, ransacReprojThreshold=thresh)

        return ptsL, ptsR, H, status

    def make(self, show_process=False, show_result=True, use_custom=False, use_mod=True):
        cap = cv2.VideoCapture(self.file_name)
        last_x = 0

        ret, total_image = cap.read()

        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))):
            ret, frame = cap.read()

            if not ret:
                break

            keypointsL, featuresL = self._FindFeatures(total_image)
            keypointsR, featuresR = self._FindFeatures(frame)


            ptsL, ptsR, H, status = self._FindTransform(keypointsR, keypointsL, featuresR, featuresL,
                                                        use_custom=use_custom, use_mod=use_mod)

            if H is not None:
                if show_process:
                    cpy = frame.copy()

                up_and_down = np.array([[[frame.shape[1], 0], [frame.shape[1], frame.shape[0]]]], dtype=np.float)
                after_points = cv2.perspectiveTransform(up_and_down, H)
                new_x = int(min(after_points[0][0][0], after_points[0][1][0]))

                if new_x - last_x < self.shift:
                    continue

                last_x = new_x

                frame = cv2.warpPerspective(frame, H, (frame.shape[1] + total_image.shape[1], frame.shape[0]))
                mask = np.ones(frame.shape, dtype=frame.dtype) * 255
                mask = cv2.warpPerspective(mask, H, (frame.shape[1] + total_image.shape[1], frame.shape[0]))
                mask = (255 - mask[0: total_image.shape[0], 0: total_image.shape[1]]).astype(np.float) / 255

                interlap_cpy = frame[0: total_image.shape[0], 0: total_image.shape[1]].astype(np.float)
                interlap_cpy += (total_image.astype(np.float) + mask * total_image.astype(np.float))
                interlap_cpy /= 2
                frame[0: total_image.shape[0], 0: total_image.shape[1]] = interlap_cpy.astype(np.uint8)
                total_image = frame[:, 0: new_x]

                while show_process:
                    # colours = sns.color_palette(n_colors=status.size)

                    # colour_count = 0
                    # for i in range(status.size):
                    #     if status[i]:
                    #         cv2.circle(cpy, (int(ptsL[i][0]), int(ptsL[i][1])), 2, colours[i], thickness=-1)
                    #         cv2.circle(frame, (int(ptsR[i][0]), int(ptsR[i][1])), 2, colours[i], thickness=-1)
                    #         colour_count += 1

                    # real_mask = np.ones(frame.shape, dtype=frame.dtype) * 255
                    # real_mask = cv2.warpPerspective(real_mask, H, (frame.shape[1] + total_image.shape[1], frame.shape[0]))
                    # real_mask = real_mask[0: total_image.shape[0], 0: total_image.shape[1]]

                    # cv2.imshow('right', cpy)
                    cv2.imshow('total', total_image)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

        cv2.imwrite('result.png', total_image)

        while show_result:
            cv2.imshow('Panorama', total_image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

#%%
if __name__ == '__main__':
    panorama = Panorama("vid.mp4", shift=100)
    panorama.make(show_process=False, show_result=True, use_custom=True, use_mod=True)