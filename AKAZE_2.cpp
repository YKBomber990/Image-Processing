#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <stdio.h>
//#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
	// 2�摜�̓ǂݍ���
	Mat image_src1 = imread("images/����.png", IMREAD_GRAYSCALE);
	Mat image_src2 = imread("images/����(�E).png", IMREAD_GRAYSCALE);

	// �����_�����o��, �L�q�q���v�Z
	vector<KeyPoint> keypoint1, keypoint2;
	Mat descriptor1, descriptor2;
	auto detector = AKAZE::create(); // BRISK, CRB, KAZE, AKAZE
	detector->detectAndCompute(image_src1, Mat(), keypoint1, descriptor1);
	detector->detectAndCompute(image_src2, Mat(), keypoint2, descriptor2);

	// �}�b�`���O���̂�
	BFMatcher matcher(NORM_HAMMING, true); // BRISK, CRB, AKAZE -- NORM_HAMMING KAZE -- NORM_L2
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);

	// �ǂ��}�b�`���O��I��
	Mat image_good_matches;
	const float threshold = 75.0f;
	vector<DMatch> good_matches;
	for (auto it = matches.begin(); it != matches.end(); ++it)
	{
		if (it->distance < threshold)
			good_matches.push_back(*it);
	}

	// �ǂ��}�b�`���O��`��
	drawMatches(image_src1, keypoint1, image_src2, keypoint2, good_matches, image_good_matches);
	imshow("AKAZE�}�b�`���O�摜(threshold = 90)", image_good_matches);
	imwrite("images/��_�����_.jpg", image_good_matches);

	waitKey();
	destroyAllWindows();

	return 0;
}