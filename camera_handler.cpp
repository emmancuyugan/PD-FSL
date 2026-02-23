#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/packet.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/graphs/holistic/holistic_landmark_cpu.pb.h>

using namespace mediapipe;

// ============================================
// Structures for landmark data
// ============================================
struct KeyPoint {
    float x, y, z, visibility;
};

struct FrameData {
    std::vector<KeyPoint> poseLandmarks;      // 33 pose points
    std::vector<KeyPoint> leftHandLandmarks;  // 21 hand points
    std::vector<KeyPoint> rightHandLandmarks; // 21 hand points
    std::vector<KeyPoint> faceLandmarks;      // 468 face points
    uint8_t* frameBuffer;                     // Raw JPEG frame
    int frameSize;
    int64_t timestamp;
};

// ============================================
// Camera Handler Class
// ============================================
class MediaPipeCameraHandler {
private:
    cv::VideoCapture camera;
    std::unique_ptr<CalculatorGraph> graph;
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::atomic<bool> isRunning{false};
    std::atomic<bool> isInitialized{false};
    std::thread processingThread;
    
    int cameraWidth = 480;
    int cameraHeight = 360;
    int cameraFps = 30;
    
public:
    MediaPipeCameraHandler() : frameQueue(), processingThread() {}
    
    ~MediaPipeCameraHandler() {
        stop();
    }
    
    // ============================================
    // Initialize Mediapipe graph
    // ============================================
    bool initializeMediaPipe() {
        std::string calculator_graph_config_contents = R"(
            input_stream: "input_video"
            output_stream: "pose_landmarks"
            output_stream: "left_hand_landmarks"
            output_stream: "right_hand_landmarks"
            output_stream: "face_landmarks"
            
            node {
              calculator: "HolisticLandmarkCpu"
              input_stream: "IMAGE:input_video"
              output_stream: "POSE_LANDMARKS:pose_landmarks"
              output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
              output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
              output_stream: "FACE_LANDMARKS:face_landmarks"
              node_options: {
                [type.googleapis.com/mediapipe.HolisticLandmarkCpuOptions] {
                  model_complexity: 1
                  smooth_landmarks: true
                  refine_face_landmarks: false
                  min_detection_confidence: 0.70
                  min_tracking_confidence: 0.70
                }
              }
            }
        )";
        
        ::mediapipe::CalculatorGraphConfig config;
        if (!::mediapipe::ParseTextProtoOrDie(calculator_graph_config_contents, &config)) {
            std::cerr << "Failed to parse graph config" << std::endl;
            return false;
        }
        
        graph = std::make_unique<CalculatorGraph>();
        auto statusOrPointers = graph->Initialize(config);
        if (!statusOrPointers.ok()) {
            std::cerr << "Failed to initialize graph: " << statusOrPointers.status() << std::endl;
            return false;
        }
        
        isInitialized = true;
        return true;
    }
    
    // ============================================
    // Initialize camera
    // ============================================
    bool initializeCamera(int deviceId = 0) {
        camera.open(deviceId);
        if (!camera.isOpened()) {
            std::cerr << "Failed to open camera" << std::endl;
            return false;
        }
        
        // Set resolution and FPS
        camera.set(cv::CAP_PROP_FRAME_WIDTH, cameraWidth);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, cameraHeight);
        camera.set(cv::CAP_PROP_FPS, cameraFps);
        camera.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Minimize buffer lag
        
        std::cout << "Camera initialized: " << cameraWidth << "x" << cameraHeight 
                  << " @ " << cameraFps << "fps" << std::endl;
        return true;
    }
    
    // ============================================
    // Extract landmarks to float array
    // ============================================
    std::vector<float> extractLandmarks(const ::mediapipe::NormalizedLandmarkList& landmarks) {
        std::vector<float> output;
        for (const auto& lm : landmarks.landmark()) {
            output.push_back(lm.x());
            output.push_back(lm.y());
            output.push_back(lm.z());
            output.push_back(lm.visibility());
        }
        return output;
    }
    
    // ============================================
    // Process frame (blocking)
    // ============================================
    bool processFrame(cv::Mat& frame, FrameData& outData) {
        if (!graph) return false;
        
        auto imageFrame = std::make_unique<ImageFrame>(
            ImageFormat::SRGB, frame.cols, frame.rows,
            frame.step, frame.data, [](uint8_t*) {}  // No-op deleter
        );
        
        int64_t now = std::chrono::system_clock::now().time_since_epoch().count();
        auto input_packet = MakePacket<ImageFrame>(std::move(imageFrame)).At(Timestamp(now));
        
        if (!graph->AddPacketToInputStream("input_video", input_packet).ok()) {
            std::cerr << "Failed to add packet" << std::endl;
            return false;
        }
        
        // Get outputs
        if (!graph->WaitUntilIdle().ok()) {
            std::cerr << "Graph error" << std::endl;
            return false;
        }
        
        auto pose_packet = graph->GetOutputStreamPoller("pose_landmarks").Next();
        auto left_hand_packet = graph->GetOutputStreamPoller("left_hand_landmarks").Next();
        auto right_hand_packet = graph->GetOutputStreamPoller("right_hand_landmarks").Next();
        auto face_packet = graph->GetOutputStreamPoller("face_landmarks").Next();
        
        outData.timestamp = now;
        
        if (pose_packet.ok()) {
            auto& landmarks = pose_packet.Value().Get<NormalizedLandmarkList>();
            for (const auto& lm : landmarks.landmark()) {
                KeyPoint kp{lm.x(), lm.y(), lm.z(), lm.visibility()};
                outData.poseLandmarks.push_back(kp);
            }
        }
        
        if (left_hand_packet.ok()) {
            auto& landmarks = left_hand_packet.Value().Get<NormalizedLandmarkList>();
            for (const auto& lm : landmarks.landmark()) {
                KeyPoint kp{lm.x(), lm.y(), lm.z(), lm.visibility()};
                outData.leftHandLandmarks.push_back(kp);
            }
        }
        
        if (right_hand_packet.ok()) {
            auto& landmarks = right_hand_packet.Value().Get<NormalizedLandmarkList>();
            for (const auto& lm : landmarks.landmark()) {
                KeyPoint kp{lm.x(), lm.y(), lm.z(), lm.visibility()};
                outData.rightHandLandmarks.push_back(kp);
            }
        }
        
        if (face_packet.ok()) {
            auto& landmarks = face_packet.Value().Get<NormalizedLandmarkList>();
            for (const auto& lm : landmarks.landmark()) {
                KeyPoint kp{lm.x(), lm.y(), lm.z(), lm.visibility()};
                outData.faceLandmarks.push_back(kp);
            }
        }
        
        // Encode frame to JPEG
        std::vector<int> jpegParams = {cv::IMWRITE_JPEG_QUALITY, 60};
        std::vector<uchar> jpegBuffer;
        cv::imencode(".jpg", frame, jpegBuffer, jpegParams);
        
        outData.frameSize = jpegBuffer.size();
        outData.frameBuffer = new uint8_t[jpegBuffer.size()];
        std::memcpy(outData.frameBuffer, jpegBuffer.data(), jpegBuffer.size());
        
        return true;
    }
    
    // ============================================
    // Main processing loop (runs in thread)
    // ============================================
    void processingLoop() {
        cv::Mat frame;
        while (isRunning) {
            if (!camera.read(frame)) {
                std::cerr << "Failed to read frame" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            FrameData frameData;
            if (processFrame(frame, frameData)) {
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    // Keep queue at max 5 frames
                    if (frameQueue.size() > 5) {
                        auto& oldFrame = frameQueue.front();
                        delete[] oldFrame.frameBuffer;
                        frameQueue.pop();
                    }
                    frameQueue.push(frameData);
                }
            }
        }
    }
    
    // ============================================
    // Public API: Start processing
    // ============================================
    bool start() {
        if (!isInitialized) {
            std::cerr << "Not initialized" << std::endl;
            return false;
        }
        
        isRunning = true;
        processingThread = std::thread(&MediaPipeCameraHandler::processingLoop, this);
        return true;
    }
    
    // ============================================
    // Public API: Stop processing
    // ============================================
    void stop() {
        isRunning = false;
        if (processingThread.joinable()) {
            processingThread.join();
        }
        camera.release();
    }
    
    // ============================================
    // Public API: Get latest frame
    // ============================================
    bool getLatestFrame(FrameData& outFrame) {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (frameQueue.empty()) {
            return false;
        }
        
        // Get most recent, discard older ones
        while (frameQueue.size() > 1) {
            auto& oldFrame = frameQueue.front();
            delete[] oldFrame.frameBuffer;
            frameQueue.pop();
        }
        
        outFrame = frameQueue.front();
        frameQueue.pop();
        return true;
    }
    
    // ============================================
    // Public API: Serialize landmarks to binary
    // ============================================
    std::vector<float> serializeLandmarks(const FrameData& frame) {
        std::vector<float> output;
        
        // Add all landmarks as float triplets (x, y, z) or quadruplets with visibility
        for (const auto& lm : frame.poseLandmarks) {
            output.push_back(lm.x);
            output.push_back(lm.y);
            output.push_back(lm.z);
        }
        
        for (const auto& lm : frame.leftHandLandmarks) {
            output.push_back(lm.x);
            output.push_back(lm.y);
            output.push_back(lm.z);
        }
        
        for (const auto& lm : frame.rightHandLandmarks) {
            output.push_back(lm.x);
            output.push_back(lm.y);
            output.push_back(lm.z);
        }
        
        return output;
    }
};

// ============================================
// C Interface for Python/ctypes binding
// ============================================
extern "C" {
    // Global handler instance
    MediaPipeCameraHandler* g_handler = nullptr;
    
    // Initialize
    int camera_init(int deviceId) {
        if (g_handler != nullptr) {
            delete g_handler;
        }
        
        g_handler = new MediaPipeCameraHandler();
        
        if (!g_handler->initializeMediaPipe()) {
            std::cerr << "MediaPipe initialization failed" << std::endl;
            return 0;
        }
        
        if (!g_handler->initializeCamera(deviceId)) {
            std::cerr << "Camera initialization failed" << std::endl;
            return 0;
        }
        
        return 1;
    }
    
    // Start processing
    int camera_start() {
        if (!g_handler) return 0;
        return g_handler->start() ? 1 : 0;
    }
    
    // Stop processing
    void camera_stop() {
        if (g_handler) {
            g_handler->stop();
        }
    }
    
    // Get frame data
    // Returns: JSON-like buffer with landmarks and JPEG
    int camera_get_frame(uint8_t* output_buffer, int buffer_size, 
                        int* pose_count, int* left_hand_count, int* right_hand_count) {
        if (!g_handler) return -1;
        
        FrameData frame;
        if (!g_handler->getLatestFrame(frame)) {
            return 0;  // No frame available
        }
        
        *pose_count = frame.poseLandmarks.size();
        *left_hand_count = frame.leftHandLandmarks.size();
        *right_hand_count = frame.rightHandLandmarks.size();
        
        int offset = 0;
        int header_size = 4 * sizeof(int);  // counts header
        
        // Write counts
        if (offset + header_size > buffer_size) return -1;
        int* counts = (int*)output_buffer;
        counts[0] = *pose_count;
        counts[1] = *left_hand_count;
        counts[2] = *right_hand_count;
        counts[3] = frame.frameSize;
        offset += header_size;
        
        // Write landmarks
        for (const auto& lm : frame.poseLandmarks) {
            if (offset + sizeof(KeyPoint) > buffer_size) return -1;
            std::memcpy(&output_buffer[offset], &lm, sizeof(KeyPoint));
            offset += sizeof(KeyPoint);
        }
        
        for (const auto& lm : frame.leftHandLandmarks) {
            if (offset + sizeof(KeyPoint) > buffer_size) return -1;
            std::memcpy(&output_buffer[offset], &lm, sizeof(KeyPoint));
            offset += sizeof(KeyPoint);
        }
        
        for (const auto& lm : frame.rightHandLandmarks) {
            if (offset + sizeof(KeyPoint) > buffer_size) return -1;
            std::memcpy(&output_buffer[offset], &lm, sizeof(KeyPoint));
            offset += sizeof(KeyPoint);
        }
        
        // Write JPEG frame
        if (offset + frame.frameSize > buffer_size) return -1;
        std::memcpy(&output_buffer[offset], frame.frameBuffer, frame.frameSize);
        offset += frame.frameSize;
        
        delete[] frame.frameBuffer;
        return offset;  // Total bytes written
    }
    
    // Cleanup
    void camera_cleanup() {
        if (g_handler) {
            delete g_handler;
            g_handler = nullptr;
        }
    }
}
