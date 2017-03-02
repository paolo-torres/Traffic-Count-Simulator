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
		double CurrentDiagonalSize;
		double CurrentAspectRatio;
		bool CurrentMatchFoundOrNewBlob;
		bool StillBeingTracked;
		int NumOfConsecutiveFramesWithoutAMatch;
		Point predictedNextPosition;
		Blob(vector<Point> _contour);
		void predictNextPosition(void);
};

#endif
