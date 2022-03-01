#include <iostream>
#include <opencv2/opencv.hpp>
#include "seeta/FaceDetector.h"

using namespace std;

int main(int argc, char** argv) {
    float f;
    float FPS[16];
    int i, Fcnt = 0;
    cv::Mat frame, origin;
    // some timing
    chrono::steady_clock::time_point Tbegin, Tend;

    for (i = 0; i < 16; i++) FPS[i] = 0.0;

    int device_id = 0;
    string ModelPath = "../model/";

    seeta::ModelSetting FD_model;
    FD_model.append(ModelPath + "face_detector.csta");
    FD_model.set_device(seeta::ModelSetting::CPU);
    FD_model.set_id(device_id);
    // init
    seeta::FaceDetector fd(FD_model);

    cv::VideoCapture cap("../../data/Walks2.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }
    cout << "Start grabbing, press ESC on Live window to terminate" << endl;

    while (1) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }
        // Rotate the camera to make the image consistent with the direction of the person
        flip(frame, frame, 1);
        if (frame.channels() == 4)
        {
            cv::cvtColor(frame, frame, CV_RGBA2BGR);
        }

        origin = frame.clone();

        SeetaImageData image;
        image.height = frame.rows;
        image.width = frame.cols;
        image.channels = frame.channels();
        image.data = frame.data;

        Tbegin = chrono::steady_clock::now();

        /* parameters you can customize
         * fd.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 20);
         * fd.set(seeta::FaceDetector::Property::PROPERTY_THRESHOLD, 0.9);
         */
        auto faces = fd.detect(image);

        Tend = chrono::steady_clock::now();

        for (int i = 0; i < faces.size; i++) {
            auto& face = faces.data[i].pos;
            cv::rectangle(frame, cv::Rect(face.x, face.y, face.width, face.height), cv::Scalar(0, 255, 0), 2);
        }

        // calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
        for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f / 16), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        cv::imshow("FaceDetection", frame);
        char esc = cv::waitKey(5);
        if (esc == 27) break;
    }
    cv::destroyAllWindows();
    return 0;
}