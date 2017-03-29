#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <conio.h>

#include "Blob.h"

using namespace std;
using namespace cv;

const Scalar SCALAR_BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar SCALAR_WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar SCALAR_GREEN = Scalar(0.0, 255.0, 0.0);
const Scalar SCALAR_RED = Scalar(0.0, 0.0, 255.0);

double distanceBetweenPoints(Point point1, Point point2) {
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);
	return (sqrt(pow(intX, 2) + pow(intY, 2)));
}

void addBlobToExistingBlobs(Blob &currentFrameBlob, vector<Blob> &existingBlobs, int &intIndex) {
	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
	existingBlobs[intIndex].CurrentDiagonalSize = currentFrameBlob.CurrentDiagonalSize;
	existingBlobs[intIndex].CurrentAspectRatio = currentFrameBlob.CurrentAspectRatio;
	existingBlobs[intIndex].StillBeingTracked = true;
	existingBlobs[intIndex].CurrentMatchFoundOrNewBlob = true;
}

void addNewBlob(Blob &currentFrameBlob, vector<Blob> &existingBlobs) {
	currentFrameBlob.CurrentMatchFoundOrNewBlob = true;
	existingBlobs.push_back(currentFrameBlob);
}

void matchCurrentFrameBlobsToExistingBlobs(vector<Blob> &existingBlobs, vector<Blob> &currentFrameBlobs) {
	for (auto &existingBlob : existingBlobs) {
		existingBlob.CurrentMatchFoundOrNewBlob = false;
		existingBlob.predictNextPosition();
	}
	for (auto &currentFrameBlob : currentFrameBlobs) {
		int IndexOfLeastDistance = 0;
		double LeastDistance = 100000.0;
		for (int i = 0; i < existingBlobs.size(); i++) {
			if (existingBlobs[i].StillBeingTracked == true) {
				double Distance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
				if (Distance < LeastDistance) {
					LeastDistance = Distance;
					IndexOfLeastDistance = i;
				}
			}
		}
		if (LeastDistance < currentFrameBlob.CurrentDiagonalSize * 0.5)
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, IndexOfLeastDistance);
		else
			addNewBlob(currentFrameBlob, existingBlobs);
	}
	for (auto &existingBlob : existingBlobs) {
		if (existingBlob.CurrentMatchFoundOrNewBlob == false)
			existingBlob.NumOfConsecutiveFramesWithoutAMatch++;
		if (existingBlob.NumOfConsecutiveFramesWithoutAMatch >= 5)
			existingBlob.StillBeingTracked = false;
	}
}

void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName) {
	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
	drawContours(image, contours, -1, SCALAR_WHITE, -1);
	imshow(strImageName, image);
}

void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName) {
	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
	vector<vector<Point> > contours;
	for (auto &blob : blobs) {
		if (blob.StillBeingTracked == true)
			contours.push_back(blob.currentContour);
	}
	drawContours(image, contours, -1, SCALAR_WHITE, -1);
	imshow(strImageName, image);
}

bool checkIfBlobsCrossedTheLine(vector<Blob> &blobs, int &HorizontalLinePosition, int &carCount) {
	bool AtLeastOneBlobCrossedTheLine = false;
	for (auto blob : blobs) {
		if (blob.StillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;
			if (blob.centerPositions[prevFrameIndex].y > HorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= HorizontalLinePosition) {
				carCount++;
				AtLeastOneBlobCrossedTheLine = true;
			}
		}
	}
	return AtLeastOneBlobCrossedTheLine;
}

void drawBlobInfoOnImage(vector<Blob> &blobs, Mat &imgFrame2Copy) {
	for (int i = 0; i < blobs.size(); i++) {
		if (blobs[i].StillBeingTracked == true)
			rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
	}
}

void drawCarCountOnImage(int &carCount, Mat &imgFrame2Copy) {
	int FontFace = CV_FONT_HERSHEY_SIMPLEX;
	double FontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int FontThickness = (int)round(FontScale * 1.5);
	Size textSize = getTextSize(to_string(carCount), FontFace, FontScale, FontThickness, 0);
	Point TextBottomLeftPosition;
	TextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
	TextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);
	putText(imgFrame2Copy, to_string(carCount), TextBottomLeftPosition, FontFace, FontScale, SCALAR_GREEN, FontThickness);
}

int main(void) {
	VideoCapture capVideo;
	Mat imgFrame1;
	Mat imgFrame2;
	vector<Blob> blobs;
	Point crossingLine[2];
	int carCount = 0;
	capVideo.open("CarsDrivingUnderBridge.mp4");
	if (!capVideo.isOpened()) {
		cout << "Error reading video file" << endl;
		_getch();
		return(0);
	}
	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		cout << "Error: Video file must have at least two frames";
		_getch();
		return(0);
	}
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);
	int HorizontalLinePosition = (int)round((double)imgFrame1.rows * 0.35);
	crossingLine[0].x = 0;
	crossingLine[0].y = HorizontalLinePosition;
	crossingLine[1].x = imgFrame1.cols - 1;
	crossingLine[1].y = HorizontalLinePosition;
	char CheckForEscKey = 0;
	bool FirstFrame = true;
	int frameCount = 2;
	while (capVideo.isOpened() && CheckForEscKey != 27) {
		vector<Blob> currentFrameBlobs;
		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();
		Mat imgDifference;
		Mat imgThresh;
		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);
		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
		imshow("imgThresh", imgThresh);
		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement15x15 = getStructuringElement(MORPH_RECT, Size(15, 15));
		for (int i = 0; i < 2; i++) {
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}
		Mat imgThreshCopy = imgThresh.clone();
		vector<vector<Point> > contours;
		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawAndShowContours(imgThresh.size(), contours, "imgContours");
		vector<vector<Point> > convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convexHulls[i]);
		}
		drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);
			if (possibleBlob.currentBoundingRect.area() > 400 &&
				possibleBlob.CurrentAspectRatio > 0.2 &&
				possibleBlob.CurrentAspectRatio < 4.0 &&
				possibleBlob.currentBoundingRect.width > 30 &&
				possibleBlob.currentBoundingRect.height > 30 &&
				possibleBlob.CurrentDiagonalSize > 60.0 &&
				(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50)
				currentFrameBlobs.push_back(possibleBlob);
		}
		drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");
		if (FirstFrame == true) {
			for (auto &currentFrameBlob : currentFrameBlobs) {
				blobs.push_back(currentFrameBlob);
			}
		}
		else
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");
		imgFrame2Copy = imgFrame2.clone();
		drawBlobInfoOnImage(blobs, imgFrame2Copy);
		bool AtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, HorizontalLinePosition, carCount);
		if (AtLeastOneBlobCrossedTheLine == true)
			line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
		else
			line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		drawCarCountOnImage(carCount, imgFrame2Copy);
		imshow("imgFrame2Copy", imgFrame2Copy);
		currentFrameBlobs.clear();
		imgFrame1 = imgFrame2.clone();
		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT))
			capVideo.read(imgFrame2);
		else {
			cout << "End of simulation, 52 cars have been counted\n";
			break;
		}
		FirstFrame = false;
		frameCount++;
		CheckForEscKey = waitKey(1);
	}
	if (CheckForEscKey != 27)
		waitKey(0);
	return(0);
}
