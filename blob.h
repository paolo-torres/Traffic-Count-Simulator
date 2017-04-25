#ifndef MY_BLOB
#define MY_BLOB

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Blob {
	public:
		vector<Point> currentContour;
		Rect currentBoundingRect;
		vector<Point> centerPositions;
		double currentDiagonalSize;
		double currentAspectRatio;
		bool currentMatchFoundOrNewBlob;
		bool stillBeingTracked;
		int numOfConsecutiveFramesWithoutAMatch;
		Point predictedNextPosition;
		Blob(vector<Point> _contour);
		void predictNextPosition(void);
};

#endif